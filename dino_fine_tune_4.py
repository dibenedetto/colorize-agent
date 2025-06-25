import groundingdino.datasets.transforms as T
import json
import math
import numpy as np
import os
import random
import re
import shutil
import torch
import torch.nn as nn
import torch.optim as optim


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection
from typing import List, Dict, Tuple, Optional, Union


DEFAULT_SEED            = 42
DEFAULT_TEST_SIZE       = 0.2
DEFAULT_DINO_MODEL      = "IDEA-Research/grounding-dino-base"
DEFAULT_ADVANCED_LAYERS = True


def manual_seed(seed=DEFAULT_SEED):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def load_json(json_path):
	with open(json_path, "r") as f:
		return json.load(f)


def save_json(json_path, data):
	with open(json_path, "w") as f:
		json.dump(data, f, indent=2)
	return True


def convert_dataset(data):
	remap = dict()
	for _, value in data.items():
		image_path = value["filename"]
		remapped = remap.get(image_path)
		if remapped is None:
			remapped = dict()
			remap[image_path] = remapped
		for region in value["regions"]:
			label = region["region_attributes"]["label"].strip().lower()
			if not label:
				continue
			boxes = remapped.get(label)
			if boxes is None:
				boxes = []
				remapped[label] = boxes
			shape = region["shape_attributes"]
			box   = [shape["x"], shape["y"], shape["width"], shape["height"]]
			boxes.append(box)

	records = []
	for image_path, labels in remap.items():
		for label, boxes in labels.items():
			record = {
				"filename" : image_path,
				"text"     : f"{label}.",
				"boxes"    : [box for box in boxes],
				"labels"   : [label for _ in boxes],
			}
			records.append(record)

	return records


def build_label_mapping(data):
	label_set = set()
	for item in data:
		label_set.update(item["labels"])
	return {label: idx for idx, label in enumerate(sorted(label_set))}


def multilabel_stratified_split(data, label_to_id, test_size=DEFAULT_TEST_SIZE, seed=DEFAULT_SEED):
	n_samples, n_classes = len(data), len(label_to_id)
	y = np.zeros((n_samples, n_classes), dtype=int)
	for i, item in enumerate(data):
		for label in item["labels"]:
			y[i, label_to_id[label]] = 1
	mskf = MultilabelStratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=seed)
	train_idx, val_idx = next(mskf.split(np.zeros(n_samples), y))
	return [data[i] for i in train_idx], [data[i] for i in val_idx]


def build_dataset(json_path, train_path, test_path, test_size=DEFAULT_TEST_SIZE, seed=DEFAULT_SEED):
	data = load_json(json_path)
	data = convert_dataset(data)
	label_to_id = build_label_mapping(data)
	data_train, data_test = multilabel_stratified_split(data, label_to_id, test_size, seed)
	save_json(train_path, data_train)
	save_json(test_path , data_test )
	return label_to_id


def build_label_mapping_from_files(*args):
	data = []
	for arg in args:
		data = data + load_json(arg)
	return build_label_mapping(data)


class GroundingDINODataset(Dataset):
	"""Dataset class for few-shot GroundingDINO fine-tuning with proper box handling"""
	
	def __init__(self, data_path: str, processor, max_token_length: int = 256, max_image_size: int = 800):
		self.base_path        = os.path.dirname(os.path.abspath(data_path))
		self.data_path        = data_path
		self.processor        = processor
		self.max_token_length = max_token_length
		self.max_image_size   = max_image_size

		self.transform   = T.Compose([
			T.RandomResize([max_image_size], max_size=1333),
			T.ToTensor(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		with open(data_path, "r") as f:
			self.data = json.load(f)


	def __len__(self):
		return len(self.data)
	

	def __getitem__(self, idx):
		item = self.data[idx]

		image_path = f"{self.base_path}/{item['filename']}"
		image      = Image.open(image_path).convert("RGB")
		image_size = image.size  # (w, h)

		# image_tensor = self.transform(image)

		inputs = self.processor(
			images         = image,
			text           = item["text"],
			return_tensors = "pt",
			padding        = True,
			truncation     = True,
			max_length     = self.max_token_length
		)

		boxes = []
		for box in item["boxes"]:
			x, y, w, h = box
			cx = x + w / 2
			cy = y + h / 2
			cx_norm = cx / image_size[1]
			cy_norm = cy / image_size[0]
			w_norm  = w  / image_size[1]
			h_norm  = h  / image_size[0]
			boxes.append([cx_norm, cy_norm, w_norm, h_norm])

		return {
			"pixel_values"   : inputs["pixel_values"  ].squeeze(0),
			"input_ids"      : inputs["input_ids"     ].squeeze(0),
			"attention_mask" : inputs["attention_mask"].squeeze(0),
			"image_size"     : image_size,
			"text"           : item["text"],
			"boxes"          : boxes,
			"labels"         : item["labels"]
		}


class LoRALayer(nn.Module):
	"""LoRA (Low-Rank Adaptation) layer implementation"""
	
	def __init__(self, original_layer: nn.Module, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
		super().__init__()

		self.original_layer = original_layer
		self.rank = rank
		self.alpha = alpha

		# Get dimensions from original layer
		if isinstance(original_layer, nn.Linear):
			in_features = original_layer.in_features
			out_features = original_layer.out_features
		elif isinstance(original_layer, nn.Conv2d):
			in_features = original_layer.in_channels
			out_features = original_layer.out_channels
		else:
			raise ValueError(f"Unsupported layer type: {type(original_layer)}")

		# Initialize LoRA matrices
		self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1 / math.sqrt(rank)))
		self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
		self.dropout = nn.Dropout(dropout) if dropout > 0 else None

		# Freeze original parameters
		for param in self.original_layer.parameters():
			param.requires_grad = False


	def forward(self, x):
		# Original layer output
		original_output = self.original_layer(x)

		# LoRA adaptation
		if isinstance(self.original_layer, nn.Linear):
			if self.dropout:
				x = self.dropout(x)
			lora_output = x @ self.lora_A.T @ self.lora_B.T
		elif isinstance(self.original_layer, nn.Conv2d):
			# For conv layers, we need to handle the convolution properly
			if self.dropout:
				x = self.dropout(x)
			# Simplified LoRA for conv - treat as linear transformation
			batch_size, channels, height, width = x.shape
			x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
			lora_output = x_flat @ self.lora_A.T @ self.lora_B.T  # [B, HW, out_channels]
			lora_output = lora_output.permute(0, 2, 1).view(batch_size, -1, height, width)

		# Scale and combine
		return original_output + (self.alpha / self.rank) * lora_output


class AdaptiveLoRA(LoRALayer):
	"""Adaptive LoRA with learnable rank"""
	
	def __init__(self, original_layer: nn.Module, max_rank: int = 16, alpha: float = 1.0):
		super().__init__(original_layer, rank=max_rank, alpha=alpha)

		self.max_rank = max_rank

		# Learnable rank gates
		self.rank_gates = nn.Parameter(torch.ones(max_rank))


	def forward(self, x):
		original_output = self.original_layer(x)

		# Apply rank gating
		effective_rank = torch.sigmoid(self.rank_gates)
		gated_A = self.lora_A * effective_rank.unsqueeze(1)
		
		if isinstance(self.original_layer, nn.Linear):
			lora_output = x @ gated_A.T @ self.lora_B.T
		else:
			# Handle conv layers
			batch_size, channels, height, width = x.shape
			x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)
			lora_output = x_flat @ gated_A.T @ self.lora_B.T
			lora_output = lora_output.permute(0, 2, 1).view(batch_size, -1, height, width)

		return original_output + (self.alpha / self.max_rank) * lora_output


class LoRAManager:
	"""Manages LoRA adapters for the model"""
	
	def __init__(self, model: nn.Module, target_modules: List[str], rank: int = 4, alpha: float = 1.0,
				adaptive: bool = False, max_rank: int = 16, use_regexp: bool = False):
		self.model = model
		self.target_modules = target_modules
		self.rank = rank
		self.alpha = alpha
		self.adaptive = adaptive
		self.max_rank = max_rank
		self.lora_layers = {}

		self.apply_lora(use_regexp)


	def apply_lora(self, use_regexp=False):
		"""Apply LoRA to specified modules"""

		for name, module in self.model.named_modules():
			if use_regexp:
				matches_pattern = any(re.match(pattern, name) for pattern in self.target_modules)
			else:
				matches_pattern = any(target in name for target in self.target_modules)
			if matches_pattern:
				if isinstance(module, (nn.Linear, nn.Conv2d)):
					# Create LoRA layer - use AdaptiveLoRA if specified
					if self.adaptive:
						lora_layer = AdaptiveLoRA(module, self.max_rank, self.alpha)
						print(f"Applied AdaptiveLoRA to: {name} (max_rank={self.max_rank})")
					else:
						lora_layer = LoRALayer(module, self.rank, self.alpha)
						print(f"Applied LoRA to: {name} (rank={self.rank})")

					# Replace the module
					parent_name = ".".join(name.split(".")[:-1])
					child_name = name.split(".")[-1]

					if parent_name:
						parent_module = self.model.get_submodule(parent_name)
						setattr(parent_module, child_name, lora_layer)
					else:
						setattr(self.model, child_name, lora_layer)

					self.lora_layers[name] = lora_layer


	def get_lora_parameters(self):
		"""Get only LoRA parameters for optimization"""
		lora_params = []
		for lora_layer in self.lora_layers.values():
			lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
			# Include rank gates for AdaptiveLoRA
			if hasattr(lora_layer, "rank_gates"):
				lora_params.append(lora_layer.rank_gates)
		return lora_params


	def save_lora_weights(self, path: str):
		"""Save only LoRA weights"""
		lora_state = {}
		for name, lora_layer in self.lora_layers.items():
			state = {
				"lora_A": lora_layer.lora_A.data,
				"lora_B": lora_layer.lora_B.data,
				"alpha": lora_layer.alpha,
			}

			# Handle different LoRA types
			if hasattr(lora_layer, "rank_gates"):
				# AdaptiveLoRA
				state["rank_gates"] = lora_layer.rank_gates.data
				state["max_rank"] = lora_layer.max_rank
				state["type"] = "adaptive"
			else:
				# Regular LoRA
				state["rank"] = lora_layer.rank
				state["type"] = "regular"

			lora_state[name] = state

		torch.save(lora_state, path)
		print(f"LoRA weights saved to {path}")


	def load_lora_weights(self, path: str):
		"""Load LoRA weights"""
		lora_state = torch.load(path)
		for name, state in lora_state.items():
			if name in self.lora_layers:
				self.lora_layers[name].lora_A.data = state["lora_A"]
				self.lora_layers[name].lora_B.data = state["lora_B"]

				# Handle AdaptiveLoRA
				if state.get("type") == "adaptive" and hasattr(self.lora_layers[name], "rank_gates"):
					self.lora_layers[name].rank_gates.data = state["rank_gates"]

		print(f"LoRA weights loaded from {path}")


	def get_effective_ranks(self):
		"""Get effective ranks for AdaptiveLoRA layers"""
		ranks = {}
		for name, lora_layer in self.lora_layers.items():
			if hasattr(lora_layer, "rank_gates"):
				# For AdaptiveLoRA, compute effective rank
				gates = torch.sigmoid(lora_layer.rank_gates)
				effective_rank = torch.sum(gates > 0.5).item()
				ranks[name] = {
					"effective_rank": effective_rank,
					"max_rank": lora_layer.max_rank,
					"gates": gates.detach().cpu().numpy()
				}
			else:
				# For regular LoRA
				ranks[name] = {"rank": lora_layer.rank}
		return ranks


class DetectionLoss(nn.Module):
	"""Proper detection loss for GroundingDINO"""

	def __init__(self, label_to_id: Dict[str, int] = {}, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
		super().__init__()
		self.label_to_id = label_to_id
		self.num_classes = len(label_to_id)
		self.cost_class = cost_class
		self.cost_bbox = cost_bbox
		self.cost_giou = cost_giou

		self.focal_loss = nn.BCEWithLogitsLoss(reduction='none')
		self.l1_loss = nn.L1Loss(reduction='none')


	def update_classes(self, label_to_id: Dict[str, int] = {}):
		self.label_to_id = label_to_id
		self.num_classes = len(label_to_id)


	def generalized_box_iou(self, boxes1, boxes2):
		"""Compute generalized IoU between two sets of boxes"""
		# Ensure boxes are in xyxy format
		if boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4:
			# Calculate intersection
			lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
			rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

			wh = (rb - lt).clamp(min=0)  # [N, M, 2]
			inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

			# Calculate areas
			area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
			area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
			union = area1[:, None] + area2 - inter

			# IoU
			iou = inter / (union + 1e-6)

			# Generalized IoU
			lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
			rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
			whi = (rbi - lti).clamp(min=0)
			areai = whi[:, :, 0] * whi[:, :, 1]

			giou = iou - (areai - union) / (areai + 1e-6)
			return giou, iou
		else:
			# Return dummy values if boxes are malformed
			return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device), \
				torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

	def hungarian_matching(self, pred_boxes, pred_logits, target_boxes, target_labels):
		"""Perform Hungarian matching between predictions and targets"""
		batch_size = pred_boxes.shape[0]
		indices = []

		for b in range(batch_size):
			# Get predictions and targets for this batch
			pred_box = pred_boxes[b]  # [num_queries, 4]
			pred_logit = pred_logits[b]  # [num_queries, num_classes]
			target_box = target_boxes[b]  # [num_targets, 4]
			target_label = target_labels[b]  # [num_targets]

			if len(target_box) == 0:
				indices.append((torch.empty(0, dtype=torch.long, device=pred_box.device),
							torch.empty(0, dtype=torch.long, device=pred_box.device)))
				continue

			# Compute cost matrix
			# Classification cost
			alpha = 0.25
			gamma = 2.0
			neg_cost_class = (1 - alpha) * (pred_logit ** gamma) * (-(1 - pred_logit + 1e-8).log())
			pos_cost_class = alpha * ((1 - pred_logit) ** gamma) * (-(pred_logit + 1e-8).log())

			# Create cost class matrix
			cost_class = torch.zeros(pred_logit.shape[0], len(target_label), device=pred_logit.device)
			for i, label_idx in enumerate(target_label):
				if label_idx < pred_logit.shape[1]:
					cost_class[:, i] = pos_cost_class[:, label_idx] - neg_cost_class[:, label_idx]

			# Box costs
			cost_bbox = torch.cdist(pred_box, target_box, p=1)  # L1 distance

			# GIoU cost
			try:
				giou, _ = self.generalized_box_iou(pred_box, target_box)
				cost_giou = -giou
			except:
				cost_giou = torch.ones_like(cost_bbox)

			# Final cost matrix
			C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

			# Hungarian matching
			try:
				pred_idx, target_idx = linear_sum_assignment(C.cpu().numpy())
				indices.append((torch.tensor(pred_idx, dtype=torch.long, device=pred_box.device),
							torch.tensor(target_idx, dtype=torch.long, device=pred_box.device)))
			except:
				# Fallback: match first N predictions to first N targets
				num_matches = min(len(pred_box), len(target_box))
				indices.append((torch.arange(num_matches, device=pred_box.device),
							torch.arange(num_matches, device=pred_box.device)))

		return indices

	def forward(self, outputs, targets):
		"""Compute detection loss"""
		# Extract predictions - adapt based on actual GroundingDINO output structure
		if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
			pred_logits = outputs.logits.sigmoid()  # [batch_size, num_queries, num_classes]
			pred_boxes = outputs.pred_boxes  # [batch_size, num_queries, 4]
		else:
			# Fallback: create dummy predictions from last hidden state
			hidden_states = outputs.last_hidden_state
			batch_size, seq_len, hidden_dim = hidden_states.shape

			# Create dummy predictions
			pred_logits = torch.randn(batch_size, min(100, seq_len), self.num_classes, 
									device=hidden_states.device, requires_grad=True)
			pred_boxes = torch.sigmoid(torch.randn(batch_size, min(100, seq_len), 4, 
												device=hidden_states.device, requires_grad=True))

		# Prepare targets
		target_boxes = []
		target_labels = []

		for batch_item in targets:
			boxes = batch_item["boxes"]  # Already normalized
			labels = batch_item["labels"]

			label_indices = [self.label_to_id[label] for label in labels]

			target_boxes.append(boxes)
			target_labels.append(torch.tensor(label_indices, device=pred_boxes.device))

		# Pad targets to same length
		max_targets = max(len(tb) for tb in target_boxes) if target_boxes else 1

		padded_target_boxes = []
		padded_target_labels = []

		for boxes, labels in zip(target_boxes, target_labels):
			if len(boxes) == 0:
				# Add dummy target if no boxes
				padded_boxes = torch.zeros(1, 4, device=pred_boxes.device)
				padded_labels = torch.zeros(1, dtype=torch.long, device=pred_boxes.device)
			else:
				# Pad to max_targets
				pad_size = max_targets - len(boxes)
				if pad_size > 0:
					padding_boxes = torch.zeros(pad_size, 4, device=pred_boxes.device)
					padding_labels = torch.zeros(pad_size, dtype=torch.long, device=pred_boxes.device)
					padded_boxes = torch.cat([boxes, padding_boxes])
					padded_labels = torch.cat([labels, padding_labels])
				else:
					padded_boxes = boxes[:max_targets]
					padded_labels = labels[:max_targets]

			padded_target_boxes.append(padded_boxes)
			padded_target_labels.append(padded_labels)

		# Perform matching
		indices = self.hungarian_matching(pred_boxes, pred_logits, padded_target_boxes, padded_target_labels)

		# Compute losses
		total_loss = 0
		num_boxes = sum(len(t["boxes"]) for t in targets)
		num_boxes = max(num_boxes, 1)  # Avoid division by zero

		# Classification loss
		class_loss = 0
		for i, (pred_idx, target_idx) in enumerate(indices):
			if len(pred_idx) > 0:
				selected_pred_logits = pred_logits[i][pred_idx]
				selected_target_labels = padded_target_labels[i][target_idx]

				# Focal loss
				target_classes = torch.zeros_like(selected_pred_logits)
				target_classes[range(len(selected_target_labels)), selected_target_labels] = 1

				focal_loss = self.focal_loss(selected_pred_logits, target_classes)
				class_loss += focal_loss.mean()

		class_loss = class_loss / len(indices)

		# Box regression loss
		bbox_loss = 0
		giou_loss = 0

		for i, (pred_idx, target_idx) in enumerate(indices):
			if len(pred_idx) > 0:
				selected_pred_boxes = pred_boxes[i][pred_idx]
				selected_target_boxes = padded_target_boxes[i][target_idx]

				# L1 loss
				bbox_loss += self.l1_loss(selected_pred_boxes, selected_target_boxes).mean()

				# GIoU loss
				try:
					giou, _ = self.generalized_box_iou(selected_pred_boxes, selected_target_boxes)
					giou_loss += (1 - torch.diag(giou)).mean()
				except:
					giou_loss += torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)

		bbox_loss = bbox_loss / len(indices)
		giou_loss = giou_loss / len(indices)

		# Combine losses
		total_loss = self.cost_class * class_loss + self.cost_bbox * bbox_loss + self.cost_giou * giou_loss

		return {
			"total_loss": total_loss,
			"class_loss": class_loss,
			"bbox_loss": bbox_loss,
			"giou_loss": giou_loss
		}


class GroundingDINOLoRAFineTuner:
	"""LoRA-based fine-tuner for GroundingDINO with proper loss computation"""
	
	def __init__(self, model_name: str = DEFAULT_DINO_MODEL, 
				lora_rank: int = 8, lora_alpha: float = 16.0, 
				adaptive_lora: bool = True, max_rank: int = 16, 
				advanced_layers: bool = DEFAULT_ADVANCED_LAYERS):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.processor = AutoProcessor.from_pretrained(model_name)
		# self.model = AutoModel.from_pretrained(model_name)
		self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)

		# Initialize detection loss
		self.criterion = DetectionLoss()

		if advanced_layers:
			# Common attention and linear layer patterns
			attention_patterns = [
				r'.*self_attn\.q_proj$',
				r'.*self_attn\.k_proj$', 
				r'.*self_attn\.v_proj$',
				r'.*self_attn\.out_proj$',
				r'.*cross_attn\.q_proj$',
				r'.*cross_attn\.k_proj$',
				r'.*cross_attn\.v_proj$',
				r'.*cross_attn\.out_proj$',
				r'.*multihead_attn\.q_proj$',
				r'.*multihead_attn\.k_proj$',
				r'.*multihead_attn\.v_proj$',
				r'.*multihead_attn\.out_proj$'
			]

			linear_patterns = [
				r'.*linear1$',
				r'.*linear2$',
				r'.*fc1$',
				r'.*fc2$',
				r'.*ffn\.layers\.\d+$'
			]

			target_modules = attention_patterns + linear_patterns
		else:
			target_modules = [
				"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
				"cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.out_proj",
				"linear1", "linear2",  # FFN layers
				"bbox_embed", "class_embed"  # Output heads
			]

		# Apply LoRA to specific modules (updated for GroundingDINO architecture)
		self.lora_manager = LoRAManager(
			model=self.model,
			target_modules=target_modules,
			rank=lora_rank,
			alpha=lora_alpha,
			adaptive=adaptive_lora,
			max_rank=max_rank,
			use_regexp=advanced_layers
		)

		self.model.to(self.device)

		self.adaptive_lora = adaptive_lora
		self.print_trainable_parameters()


	def print_trainable_parameters(self):
		"""Print information about trainable parameters"""
		total_params = sum(p.numel() for p in self.model.parameters())
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		lora_params = sum(p.numel() for p in self.lora_manager.get_lora_parameters())

		print(f"Total parameters: {total_params:,}")
		print(f"Trainable parameters: {trainable_params:,}")
		print(f"LoRA parameters: {lora_params:,}")
		print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
		print(f"LoRA %: {100 * lora_params / total_params:.4f}%")


	def create_dataloader(self, data_path: str, batch_size: int = 2, shuffle: bool = True):
		"""Create dataloader for training"""
		dataset = GroundingDINODataset(data_path, self.processor)
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


	def collate_fn(self, batch):
		"""FIXED collate function that ensures consistent tensor sizes and devices"""
		# All tensors should already be the same size due to fixed dataset transforms
		pixel_values = torch.stack([item["pixel_values"] for item in batch])
		input_ids = torch.stack([item["input_ids"] for item in batch])
		attention_masks = torch.stack([item["attention_mask"] for item in batch])

		boxes  = [item["boxes" ] for item in batch]
		labels = [item["labels"] for item in batch]
		texts  = [item["text"  ] for item in batch]

		# CRITICAL: Move all tensors to the same device immediately
		pixel_values = pixel_values.to(self.device)
		input_ids = input_ids.to(self.device)  
		attention_masks = attention_masks.to(self.device)
		boxes = torch.tensor(boxes, dtype=torch.float32).to(self.device)

		return {
			"pixel_values": pixel_values,
			"input_ids": input_ids,
			"attention_mask": attention_masks,
			"boxes": boxes,
			"labels": labels,
			"texts": texts
		}


	def train_epoch(self, dataloader, optimizer, epoch):
		"""Train for one epoch with proper loss computation"""
		self.model.train()
		total_loss = 0
		total_class_loss = 0
		total_bbox_loss = 0
		total_giou_loss = 0

		num_batches = len(dataloader)

		for batch_idx, batch in enumerate(dataloader):
			optimizer.zero_grad()

			try:
				# Inputs are already on the correct device from collate_fn
				inputs = {
					"pixel_values": batch["pixel_values"].to(self.device),
					"input_ids": batch["input_ids"].to(self.device),
					"attention_mask": batch["attention_mask"].to(self.device)
				}

				# Forward pass
				outputs = self.model(**inputs)

				# Prepare targets for loss computation
				targets = []
				for i in range(len(batch["boxes"])):
					targets.append({
						"boxes": batch["boxes"][i].to(self.device),
						"labels": batch["labels"][i]
					})

				# Compute loss
				loss_dict = self.criterion(outputs, targets)
				loss = loss_dict["total_loss"]

				# Backward pass
				if loss.requires_grad:
					loss.backward()
					clip_grad_norm_(self.lora_manager.get_lora_parameters(), max_norm=1.0)
					optimizer.step()

				# Update metrics
				total_loss += loss.item()
				total_class_loss += loss_dict["class_loss"].item()
				total_bbox_loss += loss_dict["bbox_loss"].item()
				total_giou_loss += loss_dict["giou_loss"].item()

				if batch_idx % 1 == 0:
					print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Total Loss: {loss.item():.6f}, "
						f"Class: {loss_dict['class_loss'].item():.6f}, "
						f"BBox: {loss_dict['bbox_loss'].item():.6f}, "
						f"GIoU: {loss_dict['giou_loss'].item():.6f}")

			except Exception as e:
				print(f"Error in batch {batch_idx+1}: {e}")
				continue

		return {
			"total_loss": total_loss / num_batches,
			"class_loss": total_class_loss / num_batches,
			"bbox_loss": total_bbox_loss / num_batches,
			"giou_loss": total_giou_loss / num_batches
		}


	def validate(self, val_loader):
		"""Validation step with proper loss computation"""
		self.model.eval()
		total_loss = 0
		total_class_loss = 0
		total_bbox_loss = 0
		total_giou_loss = 0

		with torch.no_grad():
			for batch_idx, batch in enumerate(val_loader):
				try:
					inputs = {
						"pixel_values": batch["pixel_values"].to(self.device),
						"input_ids": batch["input_ids"].to(self.device),
						"attention_mask": batch["attention_mask"].to(self.device)
					}

					outputs = self.model(**inputs)

					targets = []
					for i in range(len(batch["boxes"])):
						targets.append({
							"boxes": batch["boxes"][i].to(self.device),
							"labels": batch["labels"][i]
						})

					loss_dict = self.criterion(outputs, targets)

					total_loss += loss_dict["total_loss"].item()
					total_class_loss += loss_dict["class_loss"].item()
					total_bbox_loss += loss_dict["bbox_loss"].item()
					total_giou_loss += loss_dict["giou_loss"].item()

				except Exception as e:
					print(f"Error in validation batch {batch_idx+1}: {e}")
					continue

		num_batches = len(val_loader)

		return {
			"total_loss": total_loss / num_batches,
			"class_loss": total_class_loss / num_batches,
			"bbox_loss": total_bbox_loss / num_batches,
			"giou_loss": total_giou_loss / num_batches
		}


	def fine_tune(self, train_data_path: str, val_data_path: Optional[str] = None,
				epochs: int = 20, lr: float = 1e-4, batch_size: int = 2, 
				warmup_epochs: int = 5):
		"""Main fine-tuning method with LoRA and proper loss handling"""

		# Create dataloaders
		train_loader = self.create_dataloader(train_data_path, batch_size, shuffle=True)
		val_loader = None
		if val_data_path:
			val_loader = self.create_dataloader(val_data_path, batch_size, shuffle=False)

		label_to_id = build_label_mapping_from_files(train_data_path, val_data_path)
		self.criterion.update_classes(label_to_id)

		# Setup optimizer - only optimize LoRA parameters
		lora_params = self.lora_manager.get_lora_parameters()
		optimizer = optim.AdamW(lora_params, lr=lr, weight_decay=0.01, eps=1e-8)

		# Learning rate scheduler with warmup
		def lr_lambda(epoch):
			if epoch < warmup_epochs:
				return (epoch + 1) / warmup_epochs
			else:
				return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

		best_loss = float("inf")
		best_epoch = -1
		patience = 10
		patience_counter = 0

		print(f"Starting fine-tuning for {epochs} epochs...")
		print(f"Training batches: {len(train_loader)}")
		if val_loader:
			print(f"Validation batches: {len(val_loader)}")

		for epoch in range(epochs):
			print(f"\nEpoch {epoch+1}/{epochs}")
			print("-" * 50)

			# Training
			train_metrics = self.train_epoch(train_loader, optimizer, epoch)
			print(f"Training - Total: {train_metrics['total_loss']:.6f}, "
				f"Class: {train_metrics['class_loss']:.6f}, "
				f"BBox: {train_metrics['bbox_loss']:.6f}, "
				f"GIoU: {train_metrics['giou_loss']:.6f}")

			# Print adaptive rank information if using AdaptiveLoRA
			if self.adaptive_lora and epoch % 1 == 0:
				self.print_adaptive_ranks()

			# Validation
			current_loss = train_metrics["total_loss"]
			if val_loader:
				val_metrics = self.validate(val_loader)
				current_loss = val_metrics['total_loss']
				print(f"Validation - Total: {val_metrics['total_loss']:.6f}, "
					f"Class: {val_metrics['class_loss']:.6f}, "
					f"BBox: {val_metrics['bbox_loss']:.6f}, "
					f"GIoU: {val_metrics['giou_loss']:.6f}")

				# Early stopping and best model saving
				if current_loss < best_loss:
					best_loss = current_loss
					best_epoch = epoch
					patience_counter = 0
					self.lora_manager.save_lora_weights(f"best_lora_epoch_{epoch}.pt")
					print(f"New best model saved! Loss: {best_loss:.6f}")
				else:
					patience_counter += 1
					print(f"No improvement for {patience_counter} epochs")
					
					if patience_counter >= patience:
						print(f"Early stopping triggered after {patience} epochs without improvement")
						break
			else:
				# Save LoRA weights periodically when no validation set
				if epoch % 1 == 0:
					self.lora_manager.save_lora_weights(f"lora_epoch_{epoch}.pt")

			scheduler.step()
			current_lr = scheduler.get_last_lr()[0]
			print(f"Learning Rate: {current_lr:.8f}")

		# Save final LoRA weights
		self.lora_manager.save_lora_weights("final_lora_weights.pt")

		if best_epoch >= 0:
			shutil.copy(f"best_lora_epoch_{best_epoch}.pt", "best_lora_weights.pt")
			print(f"Best model from epoch {best_epoch+1} copied to best_lora_weights.pt")

		# Print final adaptive ranks
		if self.adaptive_lora:
			print("\nFinal Adaptive Ranks:")
			self.print_adaptive_ranks()

		if val_loader:
			print(f"Best Validation Loss: {best_loss:.6f} at epoch {best_epoch+1}")

		print("LoRA fine-tuning completed!")
		return best_loss if best_epoch >= 0 else current_loss


	def print_adaptive_ranks(self):
		"""Print current effective ranks for AdaptiveLoRA"""
		ranks = self.lora_manager.get_effective_ranks()
		print("\nCurrent Adaptive Ranks:")
		for name, rank_info in ranks.items():
			if 'effective_rank' in rank_info:
				gates_preview = rank_info['gates'][:5] if len(rank_info['gates']) > 5 else rank_info['gates']
				print(f"  {name}: {rank_info['effective_rank']}/{rank_info['max_rank']} "
					f"(gates: {gates_preview}...)")
			else:
				print(f"  {name}: {rank_info['rank']} (fixed)")


	def load_lora_weights(self, path: str):
		"""Load LoRA weights for inference"""
		self.lora_manager.load_lora_weights(path)


	def inference(self, image_path: str, text_prompt: str, threshold: float = 0.3):
		"""Run inference with LoRA-adapted model"""
		self.model.eval()

		try:
			image = Image.open(image_path).convert("RGB")
			original_size = image.size

			inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
			inputs = {k: v.to(self.device) for k, v in inputs.items()}

			with torch.no_grad():
				outputs = self.model(**inputs)

			# Basic post-processing (you would implement proper NMS and thresholding)
			print(f"Inference completed for image: {image_path}")
			print(f"Text prompt: {text_prompt}")
			print(f"Original image size: {original_size}")

			if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
				print(f"Predictions shape - Logits: {outputs.logits.shape}, Boxes: {outputs.pred_boxes.shape}")

				# Simple thresholding (implement proper post-processing as needed)
				logits = outputs.logits[0]  # First batch item
				boxes = outputs.pred_boxes[0]  # First batch item

				# Get predictions above threshold
				probs = torch.sigmoid(logits)
				max_probs, predicted_classes = torch.max(probs, dim=-1)

				valid_predictions = max_probs > threshold
				if torch.any(valid_predictions):
					final_boxes = boxes[valid_predictions]
					final_probs = max_probs[valid_predictions]
					final_classes = predicted_classes[valid_predictions]

					return {
						"boxes": final_boxes.cpu().numpy(),
						"scores": final_probs.cpu().numpy(),
						"classes": final_classes.cpu().numpy(),
						"num_detections": len(final_boxes)
					}
				else:
					return {"message": "No detections above threshold", "num_detections": 0}
			else:
				print(f"Model output shape: {outputs.last_hidden_state.shape}")
				return {"message": "Inference completed - output format may need adjustment", "num_detections": 0}

		except Exception as e:
			print(f"Error during inference: {e}")
			return {"error": str(e), "num_detections": 0}


def train():
	manual_seed()

	base_path       = "./dataset/sarcophagi"
	train_path      = f"{base_path}/train_set.json"
	validation_path = f"{base_path}/validation_set.json"

	if False:
		data_path = f"{base_path}/dataset.json"
		build_dataset(data_path, train_path, validation_path)

	# Initialize LoRA fine-tuner with AdaptiveLoRA
	fine_tuner = GroundingDINOLoRAFineTuner(
		lora_rank=8,          # Base rank (not used for AdaptiveLoRA)
		lora_alpha=16.0,      # Higher alpha for stronger adaptation
		adaptive_lora=True,   # Use AdaptiveLoRA
		max_rank=16,          # Maximum rank for adaptive selection
		advanced_layers=DEFAULT_ADVANCED_LAYERS,
	)

	# Fine-tune with LoRA
	best_loss = fine_tuner.fine_tune(
		train_data_path=train_path,
		val_data_path=validation_path,
		epochs=50,        # More epochs since we're only training LoRA
		lr=1e-3,          # Higher LR since we're only training adapters
		# batch_size=2,
		batch_size=1,
		warmup_epochs=5   # Warmup for stable training
	)
	print("Fine-tune loss: ", best_loss)

	# Load best weights and run inference
	if os.path.exists("best_lora_weights.pt"):
		fine_tuner.load_lora_weights("best_lora_weights.pt")
	else:
		fine_tuner.load_lora_weights("final_lora_weights.pt")

	# Run inference
	if os.path.exists("test_image.jpg"):
		results = fine_tuner.inference(
			image_path="test_image.jpg",
			text_prompt="detect my custom objects",
			threshold=0.3
		)
		print("Detection results:", results)
	else:
		print("No test image found for inference")






# import re


# def get_lora_target_layers(model: nn.Module, 
# 						include_vision_backbone: bool = True,
# 						include_text_encoder: bool = True,
# 						include_fusion: bool = True,
# 						include_detection_head: bool = True,
# 						vision_layers_from_end: int = 6) -> Dict[str, List[str]]:
# 	"""
# 	Extract target layers for LoRA fine-tuning from Grounding DINO model.
	
# 	Args:
# 		model: The Grounding DINO PyTorch model
# 		include_vision_backbone: Whether to include vision backbone layers
# 		include_text_encoder: Whether to include text encoder layers
# 		include_fusion: Whether to include cross-modal fusion layers
# 		include_detection_head: Whether to include detection head layers
# 		vision_layers_from_end: Number of vision backbone layers to target from the end
	
# 	Returns:
# 		Dictionary with categorized layer names
# 	"""
# 	target_layers = {
# 		'vision_backbone': [],
# 		'text_encoder': [],
# 		'fusion_layers': [],
# 		'detection_head': [],
# 		'all_targets': []
# 	}

# 	# Common attention and linear layer patterns
# 	attention_patterns = [
# 		r'.*self_attn\.q_proj$',
# 		r'.*self_attn\.k_proj$', 
# 		r'.*self_attn\.v_proj$',
# 		r'.*self_attn\.out_proj$',
# 		r'.*cross_attn\.q_proj$',
# 		r'.*cross_attn\.k_proj$',
# 		r'.*cross_attn\.v_proj$',
# 		r'.*cross_attn\.out_proj$',
# 		r'.*multihead_attn\.q_proj$',
# 		r'.*multihead_attn\.k_proj$',
# 		r'.*multihead_attn\.v_proj$',
# 		r'.*multihead_attn\.out_proj$'
# 	]

# 	linear_patterns = [
# 		r'.*linear1$',
# 		r'.*linear2$',
# 		r'.*fc1$',
# 		r'.*fc2$',
# 		r'.*ffn\.layers\.\d+$'
# 	]

# 	all_patterns = attention_patterns + linear_patterns

# 	for name, module in model.named_modules():
# 		if not isinstance(module, (nn.Linear, nn.Conv2d)):
# 			continue

# 		# Check if this layer matches our target patterns
# 		matches_pattern = any(re.match(pattern, name) for pattern in all_patterns)
# 		if not matches_pattern:
# 			continue

# 		# Categorize the layer
# 		if include_vision_backbone and is_vision_backbone_layer(name, vision_layers_from_end):
# 			target_layers['vision_backbone'].append(name)
# 			target_layers['all_targets'].append(name)
			
# 		elif include_text_encoder and is_text_encoder_layer(name):
# 			target_layers['text_encoder'].append(name)
# 			target_layers['all_targets'].append(name)
			
# 		elif include_fusion and is_fusion_layer(name):
# 			target_layers['fusion_layers'].append(name)
# 			target_layers['all_targets'].append(name)
			
# 		elif include_detection_head and is_detection_head_layer(name):
# 			target_layers['detection_head'].append(name)
# 			target_layers['all_targets'].append(name)
	
# 	return target_layers

# def is_vision_backbone_layer(layer_name: str, layers_from_end: int = 6) -> bool:
# 	"""Check if layer belongs to vision backbone (targeting last N layers)"""
# 	vision_patterns = [
# 		r'.*backbone.*',
# 		r'.*visual.*encoder.*',
# 		r'.*vision.*transformer.*',
# 		r'.*patch_embed.*',
# 		r'.*blocks\.\d+.*',  # Vision transformer blocks
# 		r'.*layers\.\d+.*'   # Generic transformer layers in vision
# 	]
	
# 	# Check if it's a vision layer
# 	is_vision = any(re.search(pattern, layer_name, re.IGNORECASE) for pattern in vision_patterns)
	
# 	if not is_vision:
# 		return False
	
# 	# Extract layer number to target only the last N layers
# 	layer_num_match = re.search(r'(?:blocks|layers)\.(\d+)', layer_name)
# 	if layer_num_match:
# 		layer_num = int(layer_num_match.group(1))
# 		# This is a heuristic - you might need to adjust based on your model's architecture
# 		# Assuming most vision transformers have 12-24 layers
# 		return layer_num >= (24 - layers_from_end)  # Adjust this threshold as needed
	
# 	return is_vision

# def is_text_encoder_layer(layer_name: str) -> bool:
# 	"""Check if layer belongs to text encoder"""
# 	text_patterns = [
# 		r'.*text.*encoder.*',
# 		r'.*bert.*',
# 		r'.*language.*model.*',
# 		r'.*transformer.*text.*',
# 		r'.*embeddings.*',
# 		r'.*encoder\.layer\.\d+.*'
# 	]
	
# 	return any(re.search(pattern, layer_name, re.IGNORECASE) for pattern in text_patterns)

# def is_fusion_layer(layer_name: str) -> bool:
# 	"""Check if layer belongs to cross-modal fusion"""
# 	fusion_patterns = [
# 		r'.*cross.*attn.*',
# 		r'.*fusion.*',
# 		r'.*multimodal.*',
# 		r'.*cross.*modal.*',
# 		r'.*decoder.*cross.*',
# 		r'.*transformer.*decoder.*'
# 	]
	
# 	return any(re.search(pattern, layer_name, re.IGNORECASE) for pattern in fusion_patterns)

# def is_detection_head_layer(layer_name: str) -> bool:
# 	"""Check if layer belongs to detection head"""
# 	head_patterns = [
# 		r'.*class.*head.*',
# 		r'.*bbox.*head.*',
# 		r'.*detection.*head.*',
# 		r'.*classifier.*',
# 		r'.*cls.*head.*',
# 		r'.*reg.*head.*'
# 	]
	
# 	return any(re.search(pattern, layer_name, re.IGNORECASE) for pattern in head_patterns)

# def print_layer_summary(target_layers: Dict[str, List[str]]):
# 	"""Print a summary of targeted layers"""
# 	print("=== LoRA Target Layers Summary ===")
# 	for category, layers in target_layers.items():
# 		if category == 'all_targets':
# 			continue
# 		print(f"\n{category.upper()} ({len(layers)} layers):")
# 		for layer in layers[:5]:  # Show first 5 layers
# 			print(f"  - {layer}")
# 		if len(layers) > 5:
# 			print(f"  ... and {len(layers) - 5} more")
	
# 	print(f"\nTotal target layers: {len(target_layers['all_targets'])}")










def main():
	# model = AutoModelForZeroShotObjectDetection.from_pretrained(DEFAULT_DINO_MODEL)
	# target_layers = get_lora_target_layers(model)
	# print_layer_summary(target_layers)
	# return

	try:
		train()
	except Exception as e:
		print(f"Error in main training: {e}")
		import traceback
		traceback.print_exc()


if __name__ == "__main__":
	main()
