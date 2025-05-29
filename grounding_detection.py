import torch
from transformers import pipeline

from utils import DetectionResult


class GroundingDetection:

	DEFAULT_MODEL          = "IDEA-Research/grounding-dino-tiny"
	DEFAULT_BOX_THRESHOLD  = 0.2
	DEFAULT_TEXT_THRESHOLD = 0.3


	def __init__(self):
		self.cleanup()


	@property
	def is_valid(self):
		return self._detector is not None


	@property
	def model_name(self):
		return self._model


	def setup(self, model=None, api_key=None, **kwargs):
		if model is None:
			model = GroundingDetection.DEFAULT_MODEL
		
		if model == self._model\
			and api_key == self._api_key \
			and kwargs == self._kwargs \
		:
			return True

		self.cleanup()

		device   = "cuda" if torch.cuda.is_available() else "cpu"
		detector = pipeline(model=model, task="zero-shot-object-detection", device=device)

		self._model    = model
		self._api_key  = api_key
		self._kwargs   = kwargs
		self._detector = detector

		return True


	def cleanup(self):
		self._model    = None
		self._api_key  = None
		self._kwargs   = None
		self._detector = None


	def run(self, image, labels, box_threshold=None, text_threshold=None):
		if not self.is_valid:
			raise ValueError("GroundingDetector is not valid. Please call setup() first.")

		labels = [label if label.endswith(".") else label + "." for label in labels]
		if box_threshold is None:
			box_threshold = GroundingDetection.DEFAULT_BOX_THRESHOLD
		if text_threshold is None:
			text_threshold = GroundingDetection.DEFAULT_TEXT_THRESHOLD

		results = self._detector(image, candidate_labels=labels, threshold=box_threshold)
		results = [DetectionResult.from_dict(result) for result in results]

		return results


	def __call__(self, image, labels, box_threshold=None, text_threshold=None):
		return self.run(image, labels, box_threshold=box_threshold, text_threshold=text_threshold)
