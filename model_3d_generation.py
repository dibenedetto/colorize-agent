import numpy as np
import torch
from diffusers import DiffusionPipeline


class Model3DGeneration:

	DEFAULT_MODEL     = "dylanebert/LGM-full"
	DEFAULT_3D_OBJECT = "default_model.obj"


	def __init__(self):
		self.cleanup()


	@property
	def is_valid(self):
		return self._generator is not None


	@property
	def model_name(self):
		return self._model


	def setup(self, model=None, api_key=None, **kwargs):
		if model is None:
			model = Model3DGeneration.DEFAULT_MODEL
		
		if model == self._model\
			and api_key == self._api_key \
			and kwargs == self._kwargs \
		:
			return True

		self.cleanup()

		device    = "cuda" if torch.cuda.is_available() else "cpu"
		# generator = DiffusionPipeline.from_pretrained(
		# 	model,
		# 	custom_pipeline   = "dylanebert/LGM-full",
		# 	torch_dtype       = torch.float16,
		# 	trust_remote_code = True,
		# ).to(device)
		generator = None

		self._model     = model
		self._api_key   = api_key
		self._kwargs    = kwargs
		self._generator = generator

		return True


	def cleanup(self):
		self._model     = None
		self._api_key   = None
		self._kwargs    = None
		self._generator = None


	def run(self, image, path):
		if not self.is_valid:
			raise ValueError("Model3DGeneration is not valid. Please call setup() first.")

		image_np = np.array(image, dtype=np.float32) / 255.0
		result   = self._generator("", image_np)
		self._generator.save_glb(result, path)

		return path


	def __call__(self, image, path):
		return self.run(image, path)
