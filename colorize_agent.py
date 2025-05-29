import json
import numpy as np

from llm_agent             import LLMAgent
from grounding_detection   import GroundingDetector
from semantic_segmentation import SemanticSegmenter
from utils                 import annotate_image


class ColorizeAgent:

	DEFAULT_LLM_PROMPT               = LLMAgent          .DEFAULT_INSTRUCTIONS
	DEFAULT_LLM_MODEL                = LLMAgent          .DEFAULT_MODEL
	DEFAULT_LLM_TEMPERATURE          = LLMAgent          .DEFAULT_TEMPERATURE
	DEFAULT_LLM_MAX_OUT_TOKENS       = LLMAgent          .DEFAULT_MAX_OUT_TOKENS
	DEFAULT_DETECTION_MODEL          = GroundingDetector .DEFAULT_MODEL
	DEFAULT_DETECTION_BOX_THRESHOLD  = GroundingDetector .DEFAULT_BOX_THRESHOLD
	DEFAULT_DETECTION_TEXT_THRESHOLD = GroundingDetector .DEFAULT_TEXT_THRESHOLD
	DEFAULT_SEGMENTATION_MODEL       = SemanticSegmenter .DEFAULT_MODEL
	DEFAULT_SEGMENTATION_REFINEMENT  = SemanticSegmenter .DEFAULT_POLYGON_REFINEMENT
	DEFAULT_ANNOTATION               = True


	def __init__(self):
		self._llm_agent    = None
		self._detection    = None
		self._segmentation = None


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


	def setup(self,
		llm_model=None, llm_api_key=None, llm_instructions=None, llm_temperature=None, llm_max_out_tokens=None, llm_kwargs=None,
		detection_model=None, detection_api_key=None, detection_kwargs=None,
		segmentation_model=None, segmentation_api_key=None, segmentation_kwargs=None,
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
			self._detection = GroundingDetector()
		self._detection.setup(
			model   = detection_model,
			api_key = detection_api_key,
			**(detection_kwargs or {}),
		)

		if self._segmentation is None:
			self._segmentation = SemanticSegmenter()
		self._segmentation.setup(
			model   = segmentation_model,
			api_key = segmentation_api_key,
			**(segmentation_kwargs or {}),
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


	async def run(self, image, text=None, detection_threshold=None, polygon_refinement=None, annotate=None):
		annotated_image   = None
		parts             = []
		detection_results = []

		if not image or not self.is_valid:
			return annotated_image, parts, detection_results

		if True:
			if text is not None and self._llm_agent is not None and self._llm_agent.is_valid:
				parts = await self._llm_agent(text)
				parts = self._parse_parts(parts)

		if True:
			if parts and self._detection is not None and self._detection.is_valid:
				labels = [part[2] for part in parts if len(part) >= 3 and isinstance(part[2], str)]
				if detection_threshold is None:
					detection_threshold = ColorizeAgent.DEFAULT_DETECTION_THRESHOLD
				detection_results = self._detection(image, labels, detection_threshold)

		if True:
			if self._segmentation is not None and self._segmentation.is_valid:
				if polygon_refinement is None:
					polygon_refinement = ColorizeAgent.DEFAULT_SEGMENTATION_REFINEMENT
				detection_results = self._segmentation(image, detection_results, polygon_refinement)

		if True:
			if annotate is not None:
				annotate = ColorizeAgent.DEFAULT_ANNOTATION

			if annotate and detection_results is not None:
				annotated_image = annotate_image(image, detection_results)
			else:
				annotated_image = np.array(image)

		return annotated_image, parts, detection_results


	def __call__(self, image, text=None, detection_threshold=None, polygon_refinement=None, annotate=None):
		return self.run(image, text, detection_threshold, polygon_refinement, annotate)
