import json
import numpy as np

from llm_agent             import LLMAgent
from grounding_detection   import GroundingDetection
from semantic_segmentation import SemanticSegmentation
from model_3d_generation   import Model3DGeneration
from utils                 import annotate_image


class ColorizeAgent:

	DEFAULT_LLM_PROMPT               = LLMAgent             .DEFAULT_INSTRUCTIONS
	DEFAULT_LLM_MODEL                = LLMAgent             .DEFAULT_MODEL
	DEFAULT_LLM_TEMPERATURE          = LLMAgent             .DEFAULT_TEMPERATURE
	DEFAULT_LLM_MAX_OUT_TOKENS       = LLMAgent             .DEFAULT_MAX_OUT_TOKENS
	DEFAULT_DETECTION_MODEL          = GroundingDetection   .DEFAULT_MODEL
	DEFAULT_DETECTION_BOX_THRESHOLD  = GroundingDetection   .DEFAULT_BOX_THRESHOLD
	DEFAULT_DETECTION_TEXT_THRESHOLD = GroundingDetection   .DEFAULT_TEXT_THRESHOLD
	DEFAULT_SEGMENTATION_MODEL       = SemanticSegmentation .DEFAULT_MODEL
	DEFAULT_SEGMENTATION_REFINEMENT  = SemanticSegmentation .DEFAULT_REFINEMENT
	DEFAULT_ANNOTATION               = True
	DEFAULT_3D_OBJECT                = Model3DGeneration    .DEFAULT_3D_OBJECT


	def __init__(self):
		self._llm_agent    = None
		self._detection    = None
		self._segmentation = None
		self._model3d      = None


	def _parse_parts(self, response):
		parts = None
		if isinstance(response, str):
			json_start = response.find("[")
			json_end = response.rfind("]") + 1
			if json_start >= 0 and json_end > json_start:
				json_str = response[json_start:json_end]
				json_str = json_str.replace("\n", "").replace("\t", "").replace("'", '"')
				try:
					parts = json.loads(json_str)
				except json.JSONDecodeError:
					parts = None
		return parts


	@property
	def is_valid(self):
		return self._llm_agent is not None or self._detection is not None or self._segmentation is not None


	@property
	def llm_agent(self):
		return self._llm_agent


	@property
	def detection(self):
		return self._detection


	@property
	def segmentation(self):
		return self._segmentation


	@property
	def model3d(self):
		return self._model3d


	def setup(self,
		llm_model=None, llm_api_key=None, llm_instructions=None, llm_temperature=None, llm_max_out_tokens=None, llm_kwargs=None,
		detection_model=None, detection_api_key=None, detection_kwargs=None,
		segmentation_model=None, segmentation_api_key=None, segmentation_kwargs=None,
		model3d_model=None, model3d_api_key=None, model3d_kwargs=None,
	):
		if self._llm_agent is None:
			self._llm_agent = LLMAgent()
		self._llm_agent.setup(
			model          = llm_model,
			api_key        = llm_api_key,
			instructions   = llm_instructions,
			temperature    = llm_temperature,
			max_out_tokens = llm_max_out_tokens,
			**(llm_kwargs or {}),
		)

		if self._detection is None:
			self._detection = GroundingDetection()
		self._detection.setup(
			model   = detection_model,
			api_key = detection_api_key,
			**(detection_kwargs or {}),
		)

		if self._segmentation is None:
			self._segmentation = SemanticSegmentation()
		self._segmentation.setup(
			model   = segmentation_model,
			api_key = segmentation_api_key,
			**(segmentation_kwargs or {}),
		)

		if self._model3d is None:
			self._model3d = Model3DGeneration()
		self._model3d.setup(
			model   = model3d_model,
			api_key = model3d_api_key,
			**(model3d_kwargs or {}),
		)

		return True


	def cleanup(self):
		if self._llm_agent is not None:
			self._llm_agent.cleanup()
			self._llm_agent = None

		if self._detection is not None:
			self._detection.cleanup()
			self._detection = None

		if self._segmentation is not None:
			self._segmentation.cleanup()
			self._segmentation = None

		if self._model3d is not None:
			self._model3d.cleanup()
			self._model3d = None


	async def run(self, image, text=None, detection_box_threshold=None, detection_text_threshold=None, segmentation_refinement=None, annotate=None, generate_3d=None):
		annotated_image   = None
		model_3d          = None
		parts             = []
		detection_results = []

		if not image or not self.is_valid:
			return annotated_image, model_3d, parts, detection_results

		if True:
			if text is not None and self._llm_agent is not None and self._llm_agent.is_valid:
				parts = await self._llm_agent(text)
				parts = self._parse_parts(parts)

		if True:
			if parts and self._detection is not None and self._detection.is_valid:
				labels = [part[2] for part in parts if len(part) >= 3 and isinstance(part[2], str)]
				if detection_box_threshold is None:
					detection_box_threshold = ColorizeAgent.DEFAULT_DETECTION_BOX_THRESHOLD
				if detection_text_threshold is None:
					detection_text_threshold = ColorizeAgent.DEFAULT_DETECTION_TEXT_THRESHOLD
				detection_results = self._detection(image, labels, box_threshold=detection_box_threshold, text_threshold=detection_text_threshold)

		if True:
			if self._segmentation is not None and self._segmentation.is_valid:
				if segmentation_refinement is None:
					segmentation_refinement = ColorizeAgent.DEFAULT_SEGMENTATION_REFINEMENT
				detection_results = self._segmentation(image, detection_results, segmentation_refinement)

		if True:
			if annotate is None:
				annotate = ColorizeAgent.DEFAULT_ANNOTATION

			if annotate and detection_results is not None:
				annotated_image = annotate_image(image, detection_results)
			else:
				annotated_image = np.array(image)

		if True:
			model_3d = ColorizeAgent.DEFAULT_3D_OBJECT

			if generate_3d is None:
				generate_3d = ColorizeAgent.DEFAULT_3D_OBJECT

			if generate_3d and self._model3d is not None and self._model3d.is_valid:
				model_3d = self._model3d(image)

		return annotated_image, model_3d, parts, detection_results


	def __call__(self, image, text=None, detection_threshold=None, polygon_refinement=None, annotate=None):
		return self.run(image, text, detection_threshold, polygon_refinement, annotate)
