from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel

# import litellm


class LLMAgent:

	DEFAULT_INSTRUCTIONS = """
	Analyze.
	"""

	DEFAULT_MODEL          = "openai/gpt-4o"
	DEFAULT_TEMPERATURE    = 0.5
	DEFAULT_MAX_OUT_TOKENS = 1024


	def __init__(self):
		self.cleanup()


	@property
	def is_valid(self):
		return self._agent is not None


	@property
	def model_name(self):
		return self._model


	def setup(self, model=None, api_key=None, instructions=None, temperature=None, max_out_tokens=None, **kwargs):
		if model is None:
			model = LLMAgent.DEFAULT_MODEL
		if instructions is None:
			instructions = LLMAgent.DEFAULT_INSTRUCTIONS
		if temperature is None:
			temperature = LLMAgent.DEFAULT_TEMPERATURE
		if max_out_tokens is None:
			max_out_tokens = LLMAgent.DEFAULT_MAX_OUT_TOKENS

		if model == self._model \
			and api_key == self._api_key \
			and instructions == self._instructions \
			and temperature == self._temperature \
			and max_out_tokens == self._max_out_tokens \
			and kwargs == self._kwargs \
		:
			return True

		self.cleanup()

		if model.startswith("openai/"):
			model = model[len("openai/"):]
		elif model.startswith("litellm/"):
			# litellm._turn_on_debug()
			model = model[len("litellm/"):]
			model = LitellmModel(model=model, api_key=api_key)

		agent = Agent(
			name           = "LLMAgent",
			instructions   = instructions,
			model          = model,
			model_settings = ModelSettings(temperature=temperature, max_tokens=max_out_tokens),
			**kwargs,
		)

		self._model          = model
		self._api_key        = api_key
		self._instructions   = instructions
		self._temperature    = temperature
		self._max_out_tokens = max_out_tokens
		self._kwargs         = kwargs
		self._agent          = agent

		return True


	def cleanup(self):
		self._model          = None
		self._api_key        = None
		self._instructions   = None
		self._temperature    = LLMAgent.DEFAULT_TEMPERATURE
		self._max_out_tokens = LLMAgent.DEFAULT_MAX_OUT_TOKENS
		self._kwargs         = None
		self._agent          = None


	async def run(self, message):
		if not self.is_valid:
			raise ValueError("LLMAgent is not valid. Please call setup() first.")

		message = message.strip()

		try:
			result = await Runner.run(self._agent, message)
		except Exception as e:
			print(f"Error running LLMAgent: {e}")
			return None

		# sources = []
		return result.final_output #, sources


	def __call__(self, message):
		return self.run(message)
