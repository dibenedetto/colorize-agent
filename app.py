import os
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from transformers import pipeline
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple
import cv2
import json
import logging
import warnings
from dotenv import load_dotenv, dotenv_values

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*The given NumPy array is not writeable.*")

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

MAX_OUT_TOKENS = 1024

LLM_ENHANCE_PROMPT = """
I need help identifying objects and their colors in an image based on this description:

"<text>"

Expand this description to clearly identify each object and its color.
Format your response to focus only on objects and their colors, like: "red car, blue sky, green tree, brown dog".
Only include objects that are likely visible in the image.
Be specific about object names.
"""

LLM_EXTRACT_PROMPT = """
Extract objects and their associated colors from the following text. Return a JSON list of [object, color, name] triplets,
where name refers to the object, but in a way that can be detected by an image detector based on natural language descriptions.
Only include colors mentioned in the text, or if no color is specified for an object, assign it "default".

Text: "<text>"

JSON Format: 
[
["object1", "color1", "name1"],
["object2", "color2", "name2"],
...
]
"""

# Available model options
LLM_OPTIONS = {
	"Default": None,  # Will select automatically based on availability
	"OpenAI": "openai",
	"Flan-T5-XL": "google/flan-t5-xl",
	"BART-Large": "facebook/bart-large",
	"GPT-2-Large": "gpt2-large",
	"GPT-2": "gpt2",
	"Qwen-2.5": "Qwen/Qwen2.5-0.5B",
	"Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
}

GROUNDING_DINO_OPTIONS = {
	"Swin-T": {
		"config": "GroundingDINO_SwinT_OGC.cfg.py",
		"checkpoint": "groundingdino_swint_ogc.pth"
	},
	"Swin-B": {
		"config": "GroundingDINO_SwinB.cfg.py",
		"checkpoint": "groundingdino_swinb_cogcoor.pth"
	},
}

SAM_OPTIONS = {
	"ViT-H": {
		"type": "vit_h",
		"checkpoint": "sam_vit_h_4b8939.pth"
	},
	"ViT-L": {
		"type": "vit_l",
		"checkpoint": "sam_vit_l_0b3195.pth"
	},
	"ViT-B": {
		"type": "vit_b",
		"checkpoint": "sam_vit_b_01ec64.pth"
	},
}

from huggingface_hub import login
login(HF_API_KEY)

class ModelLoader:
	"""Class to handle model loading and caching"""
	def __init__(self):
		self.groundingdino = None
		self.groundingdino_config = None
		self.groundingdino_checkpoint = None
		self.sam_predictor = None
		self.sam_type = None
		self.sam_checkpoint = None
		self.llm = None
		self.llm_name = None
		
	def get_groundingdino(self, config_path=None, checkpoint_path=None):
		# If new config or checkpoint is provided, or first load
		if ((config_path and config_path != self.groundingdino_config) or 
			(checkpoint_path and checkpoint_path != self.groundingdino_checkpoint) or 
			self.groundingdino is None):
			
			# Update paths if provided
			self.groundingdino_config = config_path or self.groundingdino_config or GROUNDING_DINO_OPTIONS["Swin-T"]["config"]
			self.groundingdino_checkpoint = checkpoint_path or self.groundingdino_checkpoint or GROUNDING_DINO_OPTIONS["Swin-T"]["checkpoint"]
			
			try:
				logger.info(f"Loading GroundingDINO with config: {self.groundingdino_config} and checkpoint: {self.groundingdino_checkpoint}")
				if not os.path.isfile(self.groundingdino_config):
					torch.hub.download_url_to_file(f"https://huggingface.co/pengxian/grounding-dino/resolve/main/{self.groundingdino_config}", self.groundingdino_config)
				if not os.path.isfile(self.groundingdino_checkpoint):
					torch.hub.download_url_to_file(f"https://huggingface.co/pengxian/grounding-dino/resolve/main/{self.groundingdino_checkpoint}", self.groundingdino_checkpoint)
				self.groundingdino = load_model(self.groundingdino_config, self.groundingdino_checkpoint)
				logger.info("GroundingDINO loaded successfully")
			except Exception as e:
				logger.error(f"Error loading GroundingDINO: {e}")
				raise
				
		return self.groundingdino
		
	def get_sam_predictor(self, sam_type=None, checkpoint_path=None):
		# If new type or checkpoint is provided, or first load
		if ((sam_type and sam_type != self.sam_type) or 
			(checkpoint_path and checkpoint_path != self.sam_checkpoint) or 
			self.sam_predictor is None):
			
			# Update paths if provided
			self.sam_type = sam_type or self.sam_type or SAM_OPTIONS["ViT-H"]["type"]
			self.sam_checkpoint = checkpoint_path or self.sam_checkpoint or SAM_OPTIONS["ViT-H"]["checkpoint"]
			
			try:
				logger.info(f"Loading SAM with type: {self.sam_type} and checkpoint: {self.sam_checkpoint}")
				if not os.path.isfile(self.sam_checkpoint):
					torch.hub.download_url_to_file(f"https://dl.fbaipublicfiles.com/segment_anything/{self.sam_checkpoint}", self.sam_checkpoint)
				sam = sam_model_registry[self.sam_type](checkpoint=self.sam_checkpoint)
				sam.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
				self.sam_predictor = SamPredictor(sam)
				logger.info("SAM loaded successfully")
			except Exception as e:
				logger.error(f"Error loading SAM: {e}")
				raise
				
		return self.sam_predictor
		
	def get_llm(self, model_name=None):
		# If new model is requested or first load
		if (model_name and model_name != self.llm_name) or self.llm is None:
			self.llm_name = model_name or self.llm_name
			
			try:
				logger.info(f"Loading LLM: {self.llm_name}")
				
				# Handle OpenAI specifically
				if self.llm_name == "openai":
					try:
						from openai import OpenAI
						self.llm = "openai"
						logger.info("Using OpenAI API")
						return self.llm
					except (ImportError, Exception) as e:
						logger.warning(f"OpenAI import failed: {e}. Falling back to local models.")
				
				# Handle Hugging Face models
				if self.llm_name and self.llm_name != "openai":
					try:
						if "t5" in self.llm_name.lower() or "bart" in self.llm_name.lower():
							self.llm = pipeline("text2text-generation", model=self.llm_name)
						else:
							self.llm = pipeline("text-generation", model=self.llm_name)
						logger.info(f"Loaded LLM: {self.llm_name}")
						return self.llm
					except Exception as e:
						logger.error(f"Failed to load {self.llm_name}: {e}")
				
				# If specific model failed or no model specified, try our options in order
				if not self.llm or self.llm == "failed":
					for model_option in ["google/flan-t5-xl", "facebook/bart-large", "gpt2-large", "gpt2"]:
						try:
							if "t5" in model_option.lower() or "bart" in model_option.lower():
								self.llm = pipeline("text2text-generation", model=model_option)
							else:
								self.llm = pipeline("text-generation", model=model_option)
							self.llm_name = model_option
							logger.info(f"Loaded fallback LLM: {model_option}")
							break
						except Exception as e:
							logger.warning(f"Failed to load {model_option}: {e}")
							continue
				
				# Last resort fallback
				if not self.llm or self.llm == "failed":
					try:
						self.llm = pipeline("text-generation", model="gpt2")
						self.llm_name = "gpt2"
						logger.info("Loaded last-resort fallback model: gpt2")
					except Exception as e:
						logger.error(f"Failed to load any LLM model: {e}")
						self.llm = "failed"
					
			except Exception as e:
				logger.error(f"Error loading LLM: {e}")
				self.llm = "failed"
				
		return self.llm

# Global model loader
model_loader = ModelLoader()
loaded_llm_enhance_model = None
loaded_llm_extract_model = None

def extract_objects_and_colors(text: str, llm_model: str = None, llm_prompt: str = None) -> List[Tuple[str, Tuple[int, int, int]]]:
	"""
	Extract objects and their associated colors from text input using an LLM.
	
	Args:
		text: Input text containing object descriptions and colors
		
	Returns:
		List of tuples containing (object_name, color)
	"""
	color_object_pairs = []

	# Use LLM to extract object-color pairs
	try:
		if llm_prompt == None:
			llm_prompt = LLM_EXTRACT_PROMPT
		
		# Create a prompt for the LLM
		prompt = llm_prompt.replace("<text>", text)

		# Use Hugging Face API if available, otherwise use OpenAI
		try:
			from transformers import pipeline
			
			# Load LLM
			global loaded_llm_extract_model
			llm = model_loader.get_llm(llm_model)
			loaded_llm_extract_model = model_loader.llm_name

			if hasattr(llm, "tokenizer") and hasattr(llm.tokenizer, "model_max_length"):
				max_length = int(llm.tokenizer.model_max_length)
				max_length = min(max_length, MAX_OUT_TOKENS)
			else:
				max_length = MAX_OUT_TOKENS
				
			# Generate response
			if llm == "openai":
				try:
					from openai import OpenAI
					client = OpenAI(api_key=OPENAI_API_KEY)
					
					response = client.chat.completions.create(
						model="gpt-3.5-turbo",
						messages=[
							{"role": "system", "content": "You extract objects and their colors from text."},
							{"role": "user", "content": prompt}
						],
						max_tokens=max_length,
						temperature=0
					)
					response_text = response.choices[0].message.content.strip()
				except Exception as e:
					logger.error(f"OpenAI API error: {e}")
					raise
			else:
				response = llm(prompt, max_new_tokens=max_length, do_sample=False)
				response_text = response[0]['generated_text']
				if response_text.startswith(prompt):
					response_text = response_text[len(prompt):].strip()
			
			# Extract the JSON part
			json_start = response_text.find('[')
			json_end = response_text.rfind(']') + 1
			
			if json_start >= 0 and json_end > json_start:
				json_str = response_text[json_start:json_end]
				json_str = json_str.replace("'", '"').replace("\\n", "\n").replace("\\t", "\t")
				try:
					color_object_pairs = json.loads(json_str)
				except json.JSONDecodeError:
					logger.warning("Failed to parse JSON from LLM response")
					color_object_pairs = []
			
		except Exception as e:
			logger.error(f"Error using LLM for extraction: {e}")
	
	except Exception as e:
		logger.error(f"LLM processing error: {e}")
	
	return color_object_pairs

def detect_objects_with_groundingdino(image_path: str, text_prompt: str, dino_config=None, dino_checkpoint=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Detect objects in the image using GroundingDINO.
	
	Args:
		image_path: Path to the input image
		text_prompt: Text prompt specifying objects to detect
		dino_config: Path to GroundingDINO config
		dino_checkpoint: Path to GroundingDINO checkpoint
		
	Returns:
		Tuple of (boxes, logits, phrases)
	"""
	model = model_loader.get_groundingdino(dino_config, dino_checkpoint)
	image_source, image = load_image(image_path)
	
	boxes, logits, phrases = predict(
		model=model,
		image=image,
		caption=text_prompt,
		box_threshold=BOX_THRESHOLD,
		text_threshold=TEXT_THRESHOLD
	)
	
	return boxes, logits, phrases

def generate_masks_with_sam(image_path: str, boxes: np.ndarray, sam_type=None, sam_checkpoint=None) -> List[np.ndarray]:
	"""
	Generate masks for detected objects using Segment Anything.
	
	Args:
		image_path: Path to the input image
		boxes: Bounding boxes from GroundingDINO
		sam_type: SAM model type
		sam_checkpoint: Path to SAM checkpoint
		
	Returns:
		List of binary masks
	"""
	predictor = model_loader.get_sam_predictor(sam_type, sam_checkpoint)
	
	# Read the image
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# Set the image for SAM
	predictor.set_image(image)
	
	# Convert boxes from normalized to absolute coordinates
	H, W, _ = image.shape
	transformed_boxes = torch.clone(boxes)
	transformed_boxes[:, 0] *= W
	transformed_boxes[:, 1] *= H
	transformed_boxes[:, 2] *= W
	transformed_boxes[:, 3] *= H
	
	# Convert to format expected by SAM
	sam_boxes = torch.tensor(transformed_boxes, dtype=torch.float).cuda() if torch.cuda.is_available() else torch.tensor(transformed_boxes, dtype=torch.float)
	
	# Get masks from SAM
	masks = []
	for box in sam_boxes:
		sam_result = predictor.predict(
			box=box.cpu().numpy(),
			multimask_output=False
		)
		masks.append(sam_result[0][0])  # Take the first mask
	
	return masks

def find_color(color_name):
	default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
	return default_colors[np.random.randint(0, len(default_colors))] 

def annotate_image(image_path: str, masks: List[np.ndarray], object_colors: List[Tuple[str, str]], phrases: List[str]) -> np.ndarray:
	"""
	Annotate the image with colored masks for detected objects.
	
	Args:
		image_path: Path to the input image
		masks: Binary masks from SAM
		object_colors: List of (object_name, color) tuples
		phrases: Detected phrases from GroundingDINO
		
	Returns:
		Annotated image
	"""
	# Read the image
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# Create a copy for annotation
	annotated_image = image.copy()
	
	# Create a color map for detected phrases
	phrase_to_color = {}
	for phrase, (obj, color) in zip(phrases, object_colors * 10):  # Repeat colors if needed
		if phrase.lower() in obj.lower() or obj.lower() in phrase.lower():
			phrase_to_color[phrase] = find_color(color)
	
	# Default colors for unmatched phrases
	default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
	
	# Apply masks with colors
	overlay = annotated_image.copy()
	
	for i, (mask, phrase) in enumerate(zip(masks, phrases)):
		# Determine color for this mask
		if phrase in phrase_to_color:
			color = phrase_to_color[phrase]
		else:
			color = default_colors[i % len(default_colors)]
		
		# Apply colored mask
		colored_mask = np.zeros_like(annotated_image)
		colored_mask[mask] = color
		
		# Blend the mask with the image
		cv2.addWeighted(colored_mask, 0.5, overlay, 1, 0, overlay)
	
	# Add labels to the image
	for i, phrase in enumerate(phrases):
		# Add text labels to the image
		cv2.putText(
			overlay,
			phrase,
			(10, 30 + i * 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,
			phrase_to_color.get(phrase, default_colors[i % len(default_colors)]),
			2
		)
	
	return overlay

def enhance_text_with_llm(text: str, llm_model: str = None, llm_prompt: str = None) -> str:
	"""
	Use LLM to enhance the text input by expanding object descriptions,
	inferring missing color information, and improving object detection.
	
	Args:
		text: Original text input from user
		llm_model: Selected LLM model name
		
	Returns:
		Enhanced text input for better object detection
	"""
	try:
		global loaded_llm_enhance_model
		llm = model_loader.get_llm(llm_model)
		loaded_llm_enhance_model = model_loader.llm_name

		if llm_prompt == None:
			llm_prompt = LLM_ENHANCE_PROMPT
		
		# Create a prompt for the LLM
		prompt = llm_prompt.replace("<text>", text)
		
		if hasattr(llm, "tokenizer") and hasattr(llm.tokenizer, "model_max_length"):
			max_length = int(llm.tokenizer.model_max_length)
			max_length = min(max_length, MAX_OUT_TOKENS)
		else:
			max_length = MAX_OUT_TOKENS

		# Try different LLM APIs based on what's available
		if llm == "openai":
			try:
				from openai import OpenAI
				client = OpenAI(api_key=OPENAI_API_KEY)
				
				response = client.chat.completions.create(
					model="gpt-3.5-turbo",
					messages=[
						{"role": "system", "content": "You help identify objects and their colors in images."},
						{"role": "user", "content": prompt}
					],
					max_tokens=max_length,
					temperature=0.3
				)
				return response.choices[0].message.content.strip()
			except Exception as e:
				logger.error(f"OpenAI API error: {e}")
				return text
				
		else:
			# Text generation pipeline
			outputs = llm(prompt, max_new_tokens=max_length, do_sample=False)
			if isinstance(outputs, list) and len(outputs) > 0:
				# Extract just the newly generated text
				generated = outputs[0]['generated_text']
				if generated.startswith(prompt):
					return generated[len(prompt):].strip()
				return generated.strip()
	
		# If we get here, return the original text
		return text
		
	except Exception as e:
		logger.error(f"Error enhancing text with LLM: {e}")
		return text

def gradio_interface(image, text, llm_enhance_choice, llm_enhance_prompt, llm_extract_choice, llm_extract_prompt, dino_choice, sam_choice, threshold_box, threshold_text):
	"""
	Gradio interface function.
	
	Args:
		image: Input image
		text: Text input
		llm_enhance_choice: Selected LLM Enhance model
		llm_enhance_prompt: LLM Enhance prompt template
		llm_extract_choice: Selected LLM Extract model
		llm_extract_prompt: LLM Extract prompt template
		dino_choice: Selected GroundingDINO model
		sam_choice: Selected SAM model
		threshold_box: Box threshold value
		threshold_text: Text threshold value
		
	Returns:
		Annotated image, enhanced text, object-color pairs, detected phrases, and status message
	"""
	# Save the input image temporarily
	temp_input_path = "temp_input.jpg"
	image.save(temp_input_path)
	
	try:
		# Update thresholds
		global BOX_THRESHOLD, TEXT_THRESHOLD
		BOX_THRESHOLD = float(threshold_box)
		TEXT_THRESHOLD = float(threshold_text)
		
		# Get model configurations
		llm_enhance_model = LLM_OPTIONS.get(llm_enhance_choice)
		llm_extract_model = LLM_OPTIONS.get(llm_extract_choice)
		dino_config = GROUNDING_DINO_OPTIONS.get(dino_choice, {}).get("config")
		dino_checkpoint = GROUNDING_DINO_OPTIONS.get(dino_choice, {}).get("checkpoint")
		sam_type = SAM_OPTIONS.get(sam_choice, {}).get("type")
		sam_checkpoint = SAM_OPTIONS.get(sam_choice, {}).get("checkpoint")
		
		# Process the enhanced text
		enhanced_text = text if llm_enhance_choice == "None" else enhance_text_with_llm(text, llm_enhance_model, llm_enhance_prompt)
		logger.info(f"Enhanced text: {enhanced_text}")
		
		# Extract objects and colors from text
		object_colors = [["wings", "white"]] if llm_extract_choice == "None" else extract_objects_and_colors(enhanced_text, llm_extract_model, llm_extract_prompt)
		logger.info(f"Detected objects and colors: {object_colors}")
		if len(object_colors) == 0:
			logger.warning("No object-color pairs detected")
			return (image, enhanced_text, "", "", 
					"No object-color pairs detected. Try adjusting text or using different models.")
		
		# Format object-color pairs for display
		object_color_display = []
		for obj, color in object_colors:
			object_color_display.append(f"{obj}: {color}")
		object_color_text = "\n".join(object_color_display)
		
		# Create a prompt for GroundingDINO
		dino_prompt = ", ".join([obj for obj, _ in object_colors])
		
		# Detect objects with GroundingDINO
		boxes, logits, phrases = detect_objects_with_groundingdino(temp_input_path, dino_prompt, dino_config, dino_checkpoint)
		logger.info(f"Detected phrases: {phrases}")
		detected_phrases = ", ".join(phrases) if len(phrases) > 0 else "No objects detected"
		
		# If no objects detected, return original image
		if len(boxes) == 0:
			logger.warning("No objects detected")
			return (image, enhanced_text, object_color_text, detected_phrases, 
					"No objects detected. Try adjusting thresholds or using different models.")
		
		# Generate masks with SAM
		masks = generate_masks_with_sam(temp_input_path, boxes, sam_type, sam_checkpoint)
		
		# Annotate the image
		annotated_image = annotate_image(temp_input_path, masks, object_colors, phrases)
		
		# Convert result to PIL Image for Gradio
		result_pil = Image.fromarray(annotated_image)
		
		return (result_pil, enhanced_text, object_color_text, detected_phrases,
				f"Processing complete with {loaded_llm_enhance_model}, {loaded_llm_extract_model}, {dino_choice}, and {sam_choice}.")
	except Exception as e:
		logger.error(f"Error in gradio_interface: {e}")
		# Return original image with error text
		draw = ImageDraw.Draw(image)
		draw.text((10, 10), f"Error: {str(e)}", fill=(255, 0, 0))
		return image, text, "", "", f"Error: {str(e)}"
	finally:
		# Clean up
		if os.path.exists(temp_input_path):
			os.remove(temp_input_path)

def create_model_options_grid():
	"""f
	Creates a grid of model selection options for the Gradio interface
	"""
	with gr.Row():
		llm_enhance_choice = gr.Dropdown(
			choices=["None"]+list(LLM_OPTIONS.keys()),
			value="Default",
			label="LLM Enhance Model"
		)
	
		llm_enhance_prompt = gr.Textbox(
			lines=4, 
			value = LLM_ENHANCE_PROMPT,
			label="Enhance Prompt"
		)

	with gr.Row():
		llm_extract_choice = gr.Dropdown(
			choices=["None"]+list(LLM_OPTIONS.keys()),
			value="Default",
			label="LLM Extract Model"
		)
	
		llm_extract_prompt = gr.Textbox(
			lines=4, 
			value=LLM_EXTRACT_PROMPT,
			label="Extract Prompt"
		)

	with gr.Row():
		dino_choice = gr.Dropdown(
			choices=list(GROUNDING_DINO_OPTIONS.keys()),
			value="Swin-T",
			label="GroundingDINO Model"
		)
		
		sam_choice = gr.Dropdown(
			choices=list(SAM_OPTIONS.keys()),
			value="ViT-H",
			label="SAM Model"
		)
		
	with gr.Row():
		threshold_box = gr.Slider(
			minimum=0.05,
			maximum=0.95,
			value=BOX_THRESHOLD,
			step=0.05,
			label="Box Confidence Threshold"
		)
		
		threshold_text = gr.Slider(
			minimum=0.05,
			maximum=0.95,
			value=TEXT_THRESHOLD,
			step=0.05,
			label="Text Confidence Threshold"
		)
		
	# Return the model options for use in the Gradio interface
	return llm_enhance_choice, llm_enhance_prompt, llm_extract_choice, llm_extract_prompt, dino_choice, sam_choice, threshold_box, threshold_text

def main():
	# Initialize random number generator
	np.random.seed(42)

	# Create Gradio interface
	with gr.Blocks() as demo:
		gr.Markdown("""
		# Colorize Agent
		
		Detect and annotate objects in images based on object-color pairs extracted from text descriptions.
		""")
		
		with gr.Tabs():
			with gr.TabItem("Annotate Images"):
				with gr.Row():
					with gr.Column(scale=1):
						# Input controls
						gr.Markdown("### Input")
						input_image = gr.Image(type="pil", label="Input Image")
						input_text = gr.Textbox(
							lines=2, 
							placeholder="Describe objects to detect and their colors (e.g., 'a red car, blue sky, green trees')", 
							label="Text Description"
						)
						
						# Model selection
						llm_enhance_choice, llm_enhance_prompt, llm_extract_choice, llm_extract_prompt, dino_choice, sam_choice, threshold_box, threshold_text = create_model_options_grid()
						
						# Process button
						submit_btn = gr.Button("Process Image", variant="primary")
					
					with gr.Column(scale=1):
						# Outputs
						gr.Markdown("### Output")
						output_image = gr.Image(type="pil", label="Annotated Image")
						enhanced_text = gr.Textbox(label="Enhanced Text", interactive=False)
						object_color_text = gr.Textbox(label="Object-Color Pairs", interactive=False)
						detected_phrases = gr.Textbox(label="Detected Phrases", interactive=False)
						status_message = gr.Textbox(label="Status", interactive=False)

			with gr.TabItem("How It Works"):
				gr.Markdown("""
				## How to use:
				1. Upload an image containing the objects you want to annotate
				2. Describe the objects and their colors (e.g., "a red car, blue sky, green trees")
				3. Select your preferred models for each component:
				- **LLM**: Enhance and processes text descriptions for better object detection
				- **GroundingDINO**: Detects objects based on text prompts
				- **SAM (Segment Anything)**: Creates precise masks of detected objects
				4. Adjust confidence thresholds if needed
				5. Click "Process Image" to generate your annotated result
				
				## Model Options:
				- **LLM Options**:
				- Default: Automatically selects based on availability
				- OpenAI: Uses OpenAI API (requires API key)
				- Flan-T5-XL: Google's T5 model for text understanding
				- BART Large: Facebook's BART model
				- GPT-2 Large/GPT-2: OpenAI GPT-2 models
				
				- **GroundingDINO Options**:
				- Swin-T: Smaller, faster model
				- Swin-B: Larger, more accurate model
				
				- **SAM Options**:
				- ViT-H: Highest quality, more resource intensive
				- ViT-L: Medium quality and resources
				- ViT-B: Faster, lower resource usage
				
				## Example inputs:
				- "red car and blue sky"
				- "yellow dog playing in green grass"
				- "person wearing blue shirt next to a brown table"
				- "black cat on white sofa"
				""")
				
				gr.Markdown("""
				## Technical Details:
				
				### Component Pipeline:
				1. **Text Processing**: The LLM analyzes your text description to extract objects and associated colors
				2. **Object Detection**: GroundingDINO locates objects in the image based on the text prompt
				3. **Segmentation**: SAM generates precise masks for each detected object
				4. **Annotation**: The system overlays colored masks on the original image
				
				### Confidence Thresholds:
				- **Box Confidence**: Higher values make detection more conservative (fewer false positives)
				- **Text Confidence**: Controls how strictly text prompts match detected objects
				""")
			
		# Set up examples

		example_text_description = """
The wings of the eagle and the clipeus-bearing Erotes are painted with two shades of red: crimson-red and ochre-red, identified through analysis as hematite.
The short feathers of the Erotes were painted in the grooved areas with Egyptian blue, creating a chiaroscuro effect with the two reds.
The painting is executed with fine and elegant brushstrokes, meticulous and attentive to the details of the sculpted relief.
Tellus’ headband, the strings of Achilles’ lyre, and the tail of the centaur on the right side of the sarcophagus were painted in ochre-red.
The corner holes of the eyes of the group on the right (of the sarcophagus), depicting Chiron and Achilles, as well as those of Oceanus and Tellus, were painted in crimson-red, while the outlines of the irises of the two Erotes were rendered in ochre red; the pupils are black.
		"""
		
		examples = [
			# ["example_image.jpg", "a red car and blue sky", "Default", LLM_ENHANCE_PROMPT, "Default", LLM_EXTRACT_PROMPT, "Swin-T", "ViT-H", 0.35, 0.25],
			# ["example_image2.jpg", "yellow dog and green grass", "GPT-2", LLM_ENHANCE_PROMPT, "GPT-2", LLM_EXTRACT_PROMPT, "Swin-T", "ViT-B", 0.3, 0.2]            
			["example.jpg", example_text_description, "None", LLM_ENHANCE_PROMPT, "None", LLM_EXTRACT_PROMPT, "Swin-T", "ViT-H", 0.3, 0.2]
		]
		
		gr.Examples(
			examples=examples,
			inputs=[input_image, input_text, llm_enhance_choice, llm_enhance_prompt, llm_extract_choice, llm_extract_prompt, dino_choice, sam_choice, threshold_box, threshold_text],
			outputs=[output_image, enhanced_text, object_color_text, detected_phrases, status_message],
			fn=gradio_interface,
			cache_examples=False
		)

		# Connect the button to the function
		submit_btn.click(
			fn=gradio_interface,
			inputs=[input_image, input_text, llm_enhance_choice, llm_enhance_prompt, llm_extract_choice, llm_extract_prompt, dino_choice, sam_choice, threshold_box, threshold_text],
			outputs=[output_image, enhanced_text, object_color_text, detected_phrases, status_message],
		)
		
	# Launch the app
	demo.launch()

if __name__ == "__main__":
	main()
