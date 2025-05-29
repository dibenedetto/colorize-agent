import torch
from transformers import AutoModel, AutoProcessor

from utils import DetectionResult, get_boxes, refine_masks


class SemanticSegmenter:

	DEFAULT_MODEL              = "facebook/sam-vit-base"
	DEFAULT_POLYGON_REFINEMENT = True


	def __init__(self):
		self.cleanup()


	@property
	def is_valid(self):
		return self._segmenter is not None


	@property
	def model_name(self):
		return self._model


	def setup(self, model=None, api_key=None, **kwargs):
		if model is None:
			model = SemanticSegmenter.DEFAULT_MODEL
		
		if model == self._model\
			and api_key == self._api_key \
			and kwargs == self._kwargs \
		:
			return True

		self.cleanup()

		device    = "cuda" if torch.cuda.is_available() else "cpu"
		processor = AutoProcessor.from_pretrained(model)
		segmenter = AutoModel.from_pretrained(model).to(device)

		self._model     = model
		self._api_key   = api_key
		self._kwargs    = kwargs
		self._device    = device
		self._processor = processor
		self._segmenter = segmenter

		return True


	def cleanup(self):
		self._model     = None
		self._api_key   = None
		self._kwargs    = None
		self._device    = None
		self._processor = None
		self._segmenter = None


	def run(self, image, detection_results=None, polygon_refinement=None):
		if not self.is_valid:
			raise ValueError("SemanticSegmenter is not valid. Please call setup() first.")

		boxes   = get_boxes(detection_results) if detection_results is not None else None
		inputs  = self._processor(images=image, input_boxes=boxes, return_tensors="pt").to(self._device)

		outputs = self._segmenter(**inputs)
		masks   = self._processor.post_process_masks(
			masks=outputs.pred_masks,
			original_sizes=inputs.original_sizes,
			reshaped_input_sizes=inputs.reshaped_input_sizes
		)[0]

		if polygon_refinement is None:
			polygon_refinement = SemanticSegmenter.DEFAULT_POLYGON_REFINEMENT

		masks = refine_masks(masks, polygon_refinement)

		if not boxes:
			detection_results = [DetectionResult() for _ in range(len(masks))]

		for detection_result, mask in zip(detection_results, masks):
			detection_result.mask = mask

		return detection_results


	def __call__(self, image, detection_results, polygon_refinement):
		return self.run(image, detection_results, polygon_refinement)
