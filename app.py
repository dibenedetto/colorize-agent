import os
import numpy as np
import cv2
import gradio as gr
from dotenv import load_dotenv

from agents import WebSearchTool, set_default_openai_key

from colorize_agent import ColorizeAgent


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
	set_default_openai_key(OPENAI_API_KEY)

HF_API_KEY = os.getenv("HF_API_KEY")

LLM_SEARCH_WEB           = False
LLM_TEMPERATURE          = ColorizeAgent.DEFAULT_LLM_TEMPERATURE
LLM_MAX_OUT_TOKENS       = ColorizeAgent.DEFAULT_LLM_MAX_OUT_TOKENS
DETECTION_BOX_THRESHOLD  = ColorizeAgent.DEFAULT_DETECTION_BOX_THRESHOLD
DETECTION_TEXT_THRESHOLD = ColorizeAgent.DEFAULT_DETECTION_TEXT_THRESHOLD
SEGMENTATION_REFINEMENT  = ColorizeAgent.DEFAULT_SEGMENTATION_REFINEMENT


LLM_INSTRUCTIONS = """
You are a visual grounding assistant. Your job is to:
- Identify objects or object parts associated to material appearance (e.g., color)
- If the object or object part uses uncommon, technical, mythological, or historical terms, search the web (if needed) to understand what the terms represent
- Then create a list of triplets containing ["object", "appearance", "simplified"]  where "object" is the object, "appearance" is material appearance, and "simplified" is the simple and recognizable terms suitable for an object detector like Grounding DINO (e.g., [['Oceanus', 'metallic red', 'bearded sea god'], ['Erotes', 'crimson-red and ochre-red', 'winged love god'], ...])
- Output the list only in Python syntax, without any additional text or explanations
"""

LLM_INSTRUCTIONS = """
You are a visual grounding assistant.
Your job is to interpret the question and create a table with three columns representing:
1) The part/object associated with appearance information.
2) Its appearance information.
3) If the part/object uses uncommon, technical, mythological, or historical terms, search the web (if needed) to understand what the terms represent and use a simplified version such that i can pass it to a grounding image detector like GeroundingDino; otherwise leave it as it is.
4) Represent the table as a Python list like [["part/object", "appearance", "simplified"], ...], without any additional text or explanations.
"""

LLM_OPTIONS = {
	"Default"                  : (None, None),
	"OpenAI/GPT-4o"            : ("openai/gpt-4o", OPENAI_API_KEY),
	"Huggingface/Llama-3.1"    : ("litellm/huggingface/meta-llama/Llama-3.1-8B-Instruct", HF_API_KEY),
}

DETECTION_OPTIONS = {
	"Default"                  : (None, None),
	"IDEA/Grounding-Dino-Tiny" : ("IDEA-Research/grounding-dino-tiny", None),
}

SEGMENTATION_OPTIONS = {
	"Default"                  : (None, None),
	"Facebook/SAM-Vit-Base"    : ("facebook/sam-vit-base", None),
}


g_agent = ColorizeAgent()


def create_model_options_grid():
	with gr.Row():
		llm_choice = gr.Dropdown(
			choices=["None"]+list(LLM_OPTIONS.keys()),
			value="Default",
			label="LLM Model"
		)

		llm_temperature = gr.Slider(
			minimum=0.0,
			maximum=2.0,
			value=LLM_TEMPERATURE,
			step=0.05,
			label="Temperature"
		)

		llm_max_out_tokens = gr.Slider(
			minimum=2**8,
			maximum=2**14,
			value=LLM_MAX_OUT_TOKENS,
			step=2**8,
			label="Max Out Tokens"
		)

	with gr.Row():
		llm_instructions = gr.Textbox(
			lines=4, 
			value= str(LLM_INSTRUCTIONS).strip(),
			label="Instructions",
		)

	with gr.Row():
		detection_choice = gr.Dropdown(
			choices=list(DETECTION_OPTIONS.keys()),
			value="Default",
			label="Detection Model"
		)

		detection_box_threshold = gr.Slider(
			minimum=0.0,
			maximum=1.0,
			value=DETECTION_BOX_THRESHOLD,
			step=0.05,
			label="Box Confidence Threshold"
		)

		detection_text_threshold = gr.Slider(
			minimum=0.0,
			maximum=1.0,
			value=DETECTION_TEXT_THRESHOLD,
			step=0.05,
			label="Text Confidence Threshold"
		)

	with gr.Row():
		segmentation_choice = gr.Dropdown(
			choices=list(SEGMENTATION_OPTIONS.keys()),
			value="Default",
			label="Segmentation Model"
		)

		segmentation_refinement = gr.Checkbox(
			value=SEGMENTATION_REFINEMENT,
			label="Refinement",
		)

	return llm_choice, llm_instructions, llm_temperature, llm_max_out_tokens, detection_choice, detection_box_threshold, detection_text_threshold, segmentation_choice, segmentation_refinement


async def gradio_interface(image, text, search_web, llm_choice, llm_instructions, llm_temperature, llm_max_out_tokens, detection_choice, detection_box_threshold, detection_text_threshold, segmentation_choice, segmentation_refinement):
	try:
		llm_model                = LLM_OPTIONS.get(llm_choice)
		llm_temperature          = float(llm_temperature)
		llm_max_out_tokens       = int(llm_max_out_tokens)
		llm_kwargs               = dict(tools=[WebSearchTool(search_context_size="high"),])

		detection_model          = DETECTION_OPTIONS.get(detection_choice)
		detection_box_threshold  = float(detection_box_threshold)
		detection_text_threshold = float(detection_text_threshold)
		detection_kwargs         = None

		segmentation_model       = SEGMENTATION_OPTIONS.get(segmentation_choice)
		segmentation_kwargs      = None

		model3d_model            = None
		model3d_api_key          = None
		model3d_kwargs           = None

		g_agent.setup(
			llm_instructions=llm_instructions, llm_model=llm_model[0], llm_api_key=llm_model[1], llm_temperature=llm_temperature, llm_max_out_tokens=llm_max_out_tokens, llm_kwargs=llm_kwargs,
			detection_model=detection_model[0], detection_api_key=detection_model[1], detection_kwargs=detection_kwargs,
			segmentation_model=segmentation_model[0], segmentation_api_key=segmentation_model[1], segmentation_kwargs=segmentation_kwargs,
			model3d_model=model3d_model, model3d_api_key=model3d_api_key, model3d_kwargs=model3d_kwargs,
		)

		annotate    = True
		generate_3d = "model3d.glb"

		annotated_image, model_3d, parts, detection_results = await g_agent(image=image, text=text, detection_box_threshold=detection_box_threshold, detection_text_threshold=detection_text_threshold, segmentation_refinement=segmentation_refinement, annotate=annotate, generate_3d=generate_3d)

		for res in detection_results:
			res.label = res.label.strip().lower()
			if res.label.endswith("."):
				res.label = res.label[:-1]
		detection_results = sorted(detection_results, key=lambda x: x.label)

		image_np = np.array(image)

		labels   = [(cv2.bitwise_and(image_np, image_np, mask=res.mask), res.label) for res in detection_results]
		sources  = parts
		status   = f"Success: operating on {g_agent.llm_agent.model_name}, {g_agent.detection.model_name}, {g_agent.segmentation.model_name}."

		return annotated_image, model_3d, parts, labels, sources, status

	except Exception as e:
		status = f"Error: {e}"
		return image, None, [], [], [], status

	finally:
		pass


def search_image(image):
	if not image:
		return None

	return None


def main():
	np.random.seed(42)

	with gr.Blocks() as demo:
		gr.Markdown("""
		# Colorize Agent

		Detect and annotate objects in images based on object-color pairs extracted from text descriptions.
		""")

		with gr.Tabs():
			with gr.TabItem("Annotate Images"):
				with gr.Row():
					with gr.Column(scale=1):
						gr.Markdown("### Input")
						input_image = gr.Image(type="pil", label="Image")
						input_text = gr.Textbox(
							lines=2, 
							placeholder="Describe objects to detect and their colors (e.g., 'a red car, blue sky, green trees')", 
							label="Text Description"
						)

						with gr.Row():
							input_search_web = gr.Checkbox(
								value=LLM_SEARCH_WEB,
								label="Search Web",
							)

							output_search_web = gr.Textbox(
								lines=2, 
								placeholder="Web search description (optional)",
								label="Web Search Description"
							)

							search_web_btn = gr.Button("Search Web", variant="primary")
							search_web_btn.click(
								fn=search_image,
								inputs=[input_image],
								outputs=[output_search_web],
							)

						llm_choice, llm_instructions, llm_temperature, llm_max_out_tokens, detection_choice, detection_box_threshold, detection_text_threshold, segmentation_choice, segmentation_refinement = create_model_options_grid()

						submit_btn = gr.Button("Process Image", variant="primary")

					with gr.Column(scale=1):
						gr.Markdown("### Output")
						output_image   = gr.Image(type="pil", label="Annotated Image")
						output_3d      = gr.Model3D(label="3D Object")
						output_parts   = gr.Dataframe(label="Parts", headers=["Description", "Appearance", "Simplified"])
						# output_labels  = gr.Dataframe(label="Labels", headers=["Label", "Mask"])
						output_labels  = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto")
						output_sources = gr.Textbox(label="Sources", interactive=False)
						output_status  = gr.Textbox(label="Status", interactive=False)

			with gr.TabItem("How It Works"):
				gr.Markdown("""
				## How to use:
				1. Upload an image containing the objects you want to annotate
				2. Describe the objects and their colors (e.g., "a red car, blue sky, green trees")
				3. Select your preferred models for each component:
				- **LLM**: Enhance and processes text descriptions for better object detection
				- **Detection**: Detects objects based on text prompts
				- **Segmentation**: Creates precise masks of detected objects
				4. Adjust confidence thresholds if needed
				5. Click "Process Image" to generate your annotated result

				## Model Options:
				- **LLM Options**:
				- Default: Automatically selects based on availability

				- **Detection Options**:
				- Default: Automatically selects based on availability

				- **Segmentation Options**:
				- Default: Automatically selects based on availability

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
				2. **Object Detection**: The detection process locates objects in the image based on the text prompt
				3. **Segmentation**: The segmentation process generates precise masks for each detected object
				4. **Annotation**: The system overlays colored masks on the original image

				### Confidence Thresholds:
				- **Detection Box Confidence**: Higher values make detection more conservative (fewer false positives)
				- **Detection Text Confidence**: Controls how strictly text prompts match detected objects
				""")

		if True:
			example_image = "example.jpg"
			example_text  = """
The wings of the eagle and the clipeus-bearing Erotes are painted with two shades of red: crimson-red and ochre-red, identified through analysis as hematite.
The short feathers of the Erotes were painted in the grooved areas with Egyptian blue, creating a chiaroscuro effect with the two reds.
The painting is executed with fine and elegant brushstrokes, meticulous and attentive to the details of the sculpted relief.
Tellus’ headband, the strings of Achilles’ lyre, and the tail of the centaur on the right side of the sarcophagus were painted in ochre-red.
The corner holes of the eyes of the group on the right (of the sarcophagus), depicting Chiron and Achilles, as well as those of Oceanus and Tellus, were painted in crimson-red, while the outlines of the irises of the two Erotes were rendered in ochre red; the pupils are black.
			"""
			example_search_web = True

			examples = [
				[example_image, example_text, example_search_web, "Default", LLM_INSTRUCTIONS, LLM_TEMPERATURE, LLM_MAX_OUT_TOKENS, "Default", DETECTION_BOX_THRESHOLD, DETECTION_TEXT_THRESHOLD, "Default", SEGMENTATION_REFINEMENT],
			]

			gr.Examples(
				examples=examples,
				inputs=[input_image, input_text, input_search_web, llm_choice, llm_instructions, llm_temperature, llm_max_out_tokens, detection_choice, detection_box_threshold, detection_text_threshold, segmentation_choice, segmentation_refinement],
				outputs=[output_image, output_3d, output_parts, output_labels, output_sources, output_status],
				fn=gradio_interface,
				cache_examples=False
			)

		submit_btn.click(
			fn=gradio_interface,
			inputs=[input_image, input_text, input_search_web, llm_choice, llm_instructions, llm_temperature, llm_max_out_tokens, detection_choice, detection_box_threshold, detection_text_threshold, segmentation_choice, segmentation_refinement],
			outputs=[output_image, output_3d, output_parts, output_labels, output_sources, output_status],
		)

	demo.launch()


if __name__ == "__main__":
	main()
