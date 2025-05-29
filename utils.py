import random
from   dataclasses import dataclass
from   typing import Dict, List, Optional, Tuple, Union

import requests
import numpy as np
import torch
import cv2
from   PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class BoundingBox:
	xmin: int
	ymin: int
	xmax: int
	ymax: int

	@property
	def xyxy(self) -> List[float]:
		return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
	score: float
	label: str
	box: BoundingBox
	mask: Optional[np.array] = None

	@classmethod
	def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
		return cls(score=detection_dict['score'],
				label=detection_dict['label'],
				box=BoundingBox(xmin=detection_dict['box']['xmin'],
								ymin=detection_dict['box']['ymin'],
								xmax=detection_dict['box']['xmax'],
								ymax=detection_dict['box']['ymax']))


def annotate_image(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
	# Convert PIL Image to OpenCV format
	# image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
	image_cv2 = np.array(image)
	image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

	# Iterate over detections and add bounding boxes and masks
	for detection in detection_results:
		label = detection.label
		score = detection.score
		box = detection.box
		mask = detection.mask

		# Sample a random color for each detection
		color = np.random.randint(0, 256, size=3)

		# Draw bounding box
		cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
		cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

		# If mask is available, apply it
		if mask is not None:
			# Convert mask to uint8
			mask_uint8 = (mask * 255).astype(np.uint8)
			contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

	return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def plot_detections(
	image: Union[Image.Image, np.ndarray],
	detections: List[DetectionResult],
	save_name: Optional[str] = None
) -> None:
	annotated_image = annotate_image(image, detections)
	plt.imshow(annotated_image)
	plt.axis('off')
	if save_name:
		plt.savefig(save_name, bbox_inches='tight')
	plt.show()


def random_named_css_colors(num_colors: int) -> List[str]:
	"""
	Returns a list of randomly selected named CSS colors.

	Args:
	- num_colors (int): Number of random colors to generate.

	Returns:
	- list: List of randomly selected named CSS colors.
	"""
	# List of named CSS colors
	named_css_colors = [
		'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
		'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
		'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
		'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
		'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
		'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
		'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
		'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
		'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
		'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
		'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
		'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
		'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
		'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
		'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
		'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
		'whitesmoke', 'yellow', 'yellowgreen'
	]

	# Sample random named CSS colors
	return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
	image: np.ndarray,
	detections: List[DetectionResult],
	class_colors: Optional[Dict[str, str]] = None
) -> None:
	# If class_colors is not provided, generate random colors for each class
	if class_colors is None:
		num_detections = len(detections)
		colors = random_named_css_colors(num_detections)
		class_colors = {}
		for i in range(num_detections):
			class_colors[i] = colors[i]


	fig = px.imshow(image)

	# Add bounding boxes
	shapes = []
	annotations = []
	for idx, detection in enumerate(detections):
		label = detection.label
		box = detection.box
		score = detection.score
		mask = detection.mask

		polygon = mask_to_polygon(mask)

		fig.add_trace(go.Scatter(
			x=[point[0] for point in polygon] + [polygon[0][0]],
			y=[point[1] for point in polygon] + [polygon[0][1]],
			mode='lines',
			line=dict(color=class_colors[idx], width=2),
			fill='toself',
			name=f"{label}: {score:.2f}"
		))

		xmin, ymin, xmax, ymax = box.xyxy
		shape = [
			dict(
				type="rect",
				xref="x", yref="y",
				x0=xmin, y0=ymin,
				x1=xmax, y1=ymax,
				line=dict(color=class_colors[idx])
			)
		]
		annotation = [
			dict(
				x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
				xref="x", yref="y",
				text=f"{label}: {score:.2f}",
			)
		]

		shapes.append(shape)
		annotations.append(annotation)

	# Update layout
	button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
	button_shapes = button_shapes + [
		dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
	]
	button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

	fig.update_layout(
		xaxis=dict(visible=False),
		yaxis=dict(visible=False),
		# margin=dict(l=0, r=0, t=0, b=0),
		showlegend=True,
		updatemenus=[
			dict(
				type="buttons",
				direction="up",
				buttons=button_shapes
			)
		],
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1
		)
	)

	# Show plot
	fig.show()


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
	# Find contours in the binary mask
	contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Find the contour with the largest area
	largest_contour = max(contours, key=cv2.contourArea)

	# Extract the vertices of the contour
	polygon = largest_contour.reshape(-1, 2).tolist()

	return polygon


def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
	"""
	Convert a polygon to a segmentation mask.

	Args:
	- polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
	- image_shape (tuple): Shape of the image (height, width) for the mask.

	Returns:
	- np.ndarray: Segmentation mask with the polygon filled.
	"""
	# Create an empty mask
	mask = np.zeros(image_shape, dtype=np.uint8)

	# Convert polygon to an array of points
	pts = np.array(polygon, dtype=np.int32)

	# Fill the polygon with white color (255)
	cv2.fillPoly(mask, [pts], color=(255,))

	return mask


def load_image(image_str: str) -> Image.Image:
	if image_str.startswith("http://"):
		image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
	else:
		image = Image.open(image_str).convert("RGB")

	return image


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
	boxes = []
	for result in results:
		xyxy = result.box.xyxy
		boxes.append(xyxy)

	return [boxes]


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
	masks = masks.cpu().float()
	masks = masks.permute(0, 2, 3, 1)
	masks = masks.mean(axis=-1)
	masks = (masks > 0).int()
	masks = masks.numpy().astype(np.uint8)
	masks = list(masks)

	if polygon_refinement:
		for idx, mask in enumerate(masks):
			shape = mask.shape
			polygon = mask_to_polygon(mask)
			mask = polygon_to_mask(polygon, shape)
			masks[idx] = mask

	return masks


# def detect(
# 	image: Image.Image,
# 	labels: List[str],
# 	threshold: float = 0.3,
# 	detector_id: Optional[str] = None
# ) -> List[Dict[str, Any]]:
# 	"""
# 	Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
# 	"""
# 	device = "cuda" if torch.cuda.is_available() else "cpu"
# 	detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
# 	object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

# 	labels = [label if label.endswith(".") else label+"." for label in labels]

# 	results = object_detector(image,  candidate_labels=labels, threshold=threshold)
# 	results = [DetectionResult.from_dict(result) for result in results]

# 	return results


# def segment(
# 	image: Image.Image,
# 	detection_results: List[Dict[str, Any]],
# 	polygon_refinement: bool = False,
# 	segmenter_id: Optional[str] = None
# ) -> List[DetectionResult]:
# 	"""
# 	Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
# 	"""
# 	device = "cuda" if torch.cuda.is_available() else "cpu"
# 	segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

# 	segmentator = AutoModel.from_pretrained(segmenter_id).to(device)
# 	processor = AutoProcessor.from_pretrained(segmenter_id)

# 	boxes = get_boxes(detection_results)
# 	inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

# 	outputs = segmentator(**inputs)
# 	masks = processor.post_process_masks(
# 		masks=outputs.pred_masks,
# 		original_sizes=inputs.original_sizes,
# 		reshaped_input_sizes=inputs.reshaped_input_sizes
# 	)[0]

# 	masks = refine_masks(masks, polygon_refinement)

# 	for detection_result, mask in zip(detection_results, masks):
# 		detection_result.mask = mask

# 	return detection_results


# def grounded_segmentation(
# 	image: Union[Image.Image, str],
# 	labels: List[str],
# 	threshold: float = 0.3,
# 	polygon_refinement: bool = False,
# 	detector_id: Optional[str] = None,
# 	segmenter_id: Optional[str] = None
# ) -> Tuple[np.ndarray, List[DetectionResult]]:
# 	if isinstance(image, str):
# 		image = load_image(image)

# 	detections = detect(image, labels, threshold, detector_id)
# 	detections = segment(image, detections, polygon_refinement, segmenter_id)

# 	return np.array(image), detections


# def parse_text_for_targets(text):
# 	prompt  = "From the following sentence, extract a list of objects or object parts with associated colors and output it in the form [{\'color\': \'red\', \'target\': \'apple\'}, ...].\n"
# 	prompt += "Colors terms near objects could be an association without the standard coloring terms.\n"
# 	prompt += "Also, phrases like 'car is red' or 'flowers are violet' associate colors to objects, in this cases 'red' to 'car' and 'violet' to 'flowers', respectively.\n"
# 	prompt += "Output the list only.\n"
# 	prompt += "This is the sentence:\n"
# 	prompt += f"<sentence> \"{text}\" </sentence>"

# 	messages = [
# 		{
# 			"role"    : "system",
# 			"content" : "You are an assistant that extracts structured color-object information."
# 		},
# 		{
# 			"role"    : "user",
# 			"content" : prompt
# 		},
# 	]

# 	model = "Qwen/Qwen2.5-0.5B"
# 	model = "meta-llama/Llama-3.2-1B"
# 	model = "mistralai/Mistral-7B-Instruct-v0.1"
# 	model = "HuggingFaceH4/zephyr-7b-alpha"

# 	device = "cuda" if torch.cuda.is_available() else "cpu"
# 	pipe = pipeline("text-generation", model=model, trust_remote_code=True, device=device, max_new_tokens=1000)

# 	response = pipe(messages)
# 	print(response)

# 	# if isinstance(response, List):
# 	# 	if len(response) > 0:
# 	# 		response = response[0]
# 	# 	else:
# 	# 		return None

# 	# if isinstance(response, Dict):
# 	# 	response = response.get("generated_text")
# 	# 	if response is None:
# 	# 		return None

# 	# if isinstance(response, List):
# 	# 	if len(response) >= 3:
# 	# 		response = response[2]

# 	# if isinstance(response, Dict):
# 	# 	response = response.get("content")
# 	# 	if response is None:
# 	# 		return None

# 	# if not isinstance(response, str):
# 	# 	return None

# 	items = None
# 	try:
# 		items = response[0]["generated_text"][2]["content"]
# 	except:
# 		return None

# 	if not isinstance(items, str):
# 		return None

# 	items = items.replace("\\n", "\n").replace("\\t", "\t")

# 	result = None
# 	try:
# 		result = eval(items)
# 	except:
# 		return None

# 	return result


# def main():
# 	image_path = "example.jpg"

# 	text_description = """
# 		The wings of the eagle and the clipeus-bearing Erotes are painted with two shades of red: crimson-red and ochre-red, identified through analysis as hematite.
# 		The short feathers of the Erotes were painted in the grooved areas with Egyptian blue, creating a chiaroscuro effect with the two reds.
# 		The painting is executed with fine and elegant brushstrokes, meticulous and attentive to the details of the sculpted relief.
# 		Tellus’ headband, the strings of Achilles’ lyre, and the tail of the centaur on the right side of the sarcophagus were painted in ochre-red.
# 		The corner holes of the eyes of the group on the right (of the sarcophagus), depicting Chiron and Achilles, as well as those of Oceanus and Tellus, were painted in crimson-red, while the outlines of the irises of the two Erotes were rendered in ochre-red; the pupils are black.
# 	"""

# 	result = parse_text_for_targets(text_description)
# 	print(result)

# 	# src_file_path = "model/scene.gltf"
# 	# dst_file_path = "model/scene_clean.gltf"
# 	# data = None
# 	# with open(src_file_path, "r") as file:
# 	# 	data = json.load(file)
# 	# if "meshes" in data:
# 	# 	# data["meshes"] = [
# 	# 	# 	mesh for mesh in data["meshes"]
# 	# 	# 	if not any(keyword in mesh.get("name", "").lower() for keyword in ["window", "door", "gate"])
# 	# 	# ]

# 	# 	data["meshes"] = [
# 	# 		mesh if not any(keyword in mesh.get("name", "").lower() for keyword in ["window", "door", "gate", "tree", "leaf", "leaves"]) else dict(name=mesh["name"], primitives=[]) for mesh in data["meshes"]
# 	# 	]

# 	# 	# for mesh in data["meshes"]:
# 	# 	# 	if any(keyword in mesh.get("name", "").lower() for keyword in ["window", "door", "gate"]):
# 	# 	# 		for primitive in mesh.get("primitives", []):
# 	# 	# 			primitive["indices"] = 1

# 	# with open(dst_file_path, "w") as file:
# 	# 	json.dump(data, file, indent=2)

# 	print()
# 	print("done.")


# if __name__ == "__main__":
# 	main()
