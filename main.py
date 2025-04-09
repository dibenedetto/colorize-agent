import numpy as np
import cv2
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel, pipeline
# from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import re

class MultilingualColorReconstructionAgent:
    def __init__(self, language="english"):
        """
        Initialize the agent with specified language support
        
        Parameters:
        language: String specifying the language ("english", "latin", "french", "italian", "spanish", "german")
        """
        # Set language
        self.set_language(language)
        
        # Load CLIP model for vision-language alignment
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Setup translation pipeline for non-English languages
        if language != "english":
            self.translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{self.language_code}-en")
        else:
            self.translator = None
        
        # Setup color embeddings database
        self.color_database = self._initialize_color_database()
        
        print(f"Color Reconstruction Agent initialized successfully with {language} language support.")
    
    def set_language(self, language):
        """Set the language for text processing"""
        language = language.lower()
        supported_languages = {
            "english": {
                "code": "en",
                "model": "bert-base-uncased",
                "color_terms_file": "english_color_terms.json"
            },
            "latin": {
                "code": "la",
                "model": "pierreguillou/bert-base-multilingual-uncased-latin",
                "color_terms_file": "latin_color_terms.json"
            },
            "french": {
                "code": "fr",
                "model": "camembert-base",
                "color_terms_file": "french_color_terms.json"
            },
            "italian": {
                "code": "it",
                "model": "dbmdz/bert-base-italian-uncased",
                "color_terms_file": "italian_color_terms.json"
            },
            "spanish": {
                "code": "es",
                "model": "dccuchile/bert-base-spanish-wwm-uncased",
                "color_terms_file": "spanish_color_terms.json"
            },
            "german": {
                "code": "de",
                "model": "dbmdz/bert-base-german-uncased",
                "color_terms_file": "german_color_terms.json"
            }
        }
        
        if language not in supported_languages:
            raise ValueError(f"Language '{language}' not supported. Choose from: {', '.join(supported_languages.keys())}")
        
        self.language = language
        self.language_code = supported_languages[language]["code"]
        
        # Load language-specific language model
        model_name = supported_languages[language]["model"]
        self.text_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setup language-specific color patterns
        self._setup_language_patterns()
    
    def _setup_language_patterns(self):
        """Setup language-specific regex patterns for color extraction"""
        # Define color terms for each language
        color_terms = {
            "english": r"red|blue|green|yellow|purple|brown|black|white|gray|grey|pink|orange|gold|silver|bronze|copper|turquoise|cyan|magenta|violet|indigo|teal|maroon|navy|olive|crimson|azure|beige|coral|ivory|khaki|lavender|lilac|mauve|mint|plum|rose|rust|salmon|tan|teal|violet|amber|apricot|aqua|auburn|burgundy|chartreuse|cobalt|fuchsia|garnet|indigo|jade|lemon|lime|mahogany|mustard|ochre|periwinkle|ruby|sapphire|scarlet|sienna|slate|tangerine|topaz|umber|wheat|vermilion|verdigris|viridian",
            
            "latin": r"ruber|rubra|rubrum|purpureus|purpurea|purpureum|caeruleus|caerulea|caeruleum|viridis|viride|flavus|flava|flavum|croceus|crocea|croceum|luteus|lutea|luteum|albus|alba|album|ater|atra|atrum|niger|nigra|nigrum|candidus|candida|candidum|aureus|aurea|aureum|argenteus|argentea|argenteum|ferrugineus|ferruginea|ferrugineum|fulvus|fulva|fulvum|gilvus|gilva|gilvum|glaucus|glauca|glaucum|lividus|livida|lividum|puniceus|punicea|puniceum|roseus|rosea|roseum|sanguineus|sanguinea|sanguineum|spadix|vitreus|vitrea|vitreum",
            
            "french": r"rouge|bleu|vert|jaune|violet|marron|noir|blanc|gris|rose|orange|or|argent|bronze|cuivre|turquoise|cyan|magenta|indigo|marine|olive|cramoisi|azur|beige|corail|ivoire|kaki|lavande|lilas|mauve|menthe|prune|rouille|saumon|ocre|pourpre|bordeaux|châtain|écarlate|vermeil|vermillon",
            
            "italian": r"rosso|blu|verde|giallo|viola|marrone|nero|bianco|grigio|rosa|arancione|oro|argento|bronzo|rame|turchese|ciano|magenta|indaco|marina|oliva|cremisi|azzurro|beige|corallo|avorio|cachi|lavanda|lilla|malva|menta|prugna|ruggine|salmone|ocra|porpora|bordeaux|castano|scarlatto|vermiglio",
            
            "spanish": r"rojo|azul|verde|amarillo|morado|marrón|negro|blanco|gris|rosa|naranja|oro|plata|bronce|cobre|turquesa|cian|magenta|índigo|marino|oliva|carmesí|celeste|beige|coral|marfil|caqui|lavanda|lila|malva|menta|ciruela|óxido|salmón|ocre|púrpura|burdeos|castaño|escarlata|bermellón",
            
            "german": r"rot|blau|grün|gelb|lila|braun|schwarz|weiß|grau|pink|orange|gold|silber|bronze|kupfer|türkis|cyan|magenta|indigo|marine|oliv|karminrot|azurblau|beige|koralle|elfenbein|khaki|lavendel|flieder|malve|minze|pflaume|rost|lachs|ocker|purpur|burgunder|kastanienbraun|scharlachrot|zinnoberrot"
        }
        
        # We need language-specific patterns for verb forms and constructions
        color_pattern_templates = {
            "english": [
                r"((?:\w+\s)?(?:{colors}))\s+(?:colored\s+)?((?:\w+\s*){1,3})",
                r"((?:\w+\s*){1,3})\s+(?:is|are|was|were|appears|appeared|looks|looked)\s+(?:\w+\s)?({colors})"
            ],
            
            "latin": [
                r"((?:\w+\s)?(?:{colors}))\s+((?:\w+\s*){1,3})",
                r"((?:\w+\s*){1,3})\s+(?:est|sunt|erat|erant|videtur|videntur)\s+(?:\w+\s)?({colors})"
            ],
            
            "french": [
                r"((?:\w+\s)?(?:{colors}))\s+((?:\w+\s*){1,3})",
                r"((?:\w+\s*){1,3})\s+(?:est|sont|était|étaient|paraît|paraissent|semble|semblent)\s+(?:\w+\s)?({colors})"
            ],
            
            "italian": [
                r"((?:\w+\s)?(?:{colors}))\s+((?:\w+\s*){1,3})",
                r"((?:\w+\s*){1,3})\s+(?:è|sono|era|erano|appare|appaiono|sembra|sembrano)\s+(?:\w+\s)?({colors})"
            ],
            
            "spanish": [
                r"((?:\w+\s)?(?:{colors}))\s+((?:\w+\s*){1,3})",
                r"((?:\w+\s*){1,3})\s+(?:es|son|era|eran|aparece|aparecen|parece|parecen)\s+(?:\w+\s)?({colors})"
            ],
            
            "german": [
                r"((?:\w+\s)?(?:{colors}))\s+((?:\w+\s*){1,3})",
                r"((?:\w+\s*){1,3})\s+(?:ist|sind|war|waren|erscheint|erscheinen|sieht|sehen)\s+(?:\w+\s)?({colors})"
            ]
        }
        
        # Format patterns with color terms
        self.color_patterns = []
        for template in color_pattern_templates[self.language]:
            pattern = template.format(colors=color_terms[self.language])
            self.color_patterns.append(pattern)
    
    def _initialize_color_database(self):
        """Initialize database of color terms and their RGB values for the chosen language"""
        # Basic colors with RGB values (language-independent)
        colors_rgb = {
            # Reds
            "red": [255, 0, 0],
            "dark_red": [139, 0, 0],
            "crimson": [220, 20, 60],
            "maroon": [128, 0, 0],
            "vermilion": [227, 66, 52],
            "burgundy": [128, 0, 32],
            
            # Blues
            "blue": [0, 0, 255],
            "navy": [0, 0, 128],
            "azure": [0, 127, 255],
            "cobalt": [0, 71, 171],
            "ultramarine": [18, 10, 143],
            "cerulean": [0, 123, 167],
            
            # Greens
            "green": [0, 255, 0],
            "olive": [128, 128, 0],
            "emerald": [80, 200, 120],
            "sage": [188, 184, 138],
            "viridian": [64, 130, 109],
            "malachite": [11, 218, 81],
            
            # Yellows
            "yellow": [255, 255, 0],
            "gold": [255, 215, 0],
            "amber": [255, 191, 0],
            "ochre": [204, 119, 34],
            "mustard": [255, 219, 88],
            
            # Purples
            "purple": [128, 0, 128],
            "violet": [238, 130, 238],
            "lavender": [230, 230, 250],
            "mauve": [204, 153, 204],
            "indigo": [75, 0, 130],
            
            # Browns
            "brown": [165, 42, 42],
            "tan": [210, 180, 140],
            "sienna": [160, 82, 45],
            "umber": [99, 81, 71],
            "terracotta": [226, 114, 91],
            
            # Neutrals
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "gray": [128, 128, 128],
            "silver": [192, 192, 192],
            
            # Others
            "pink": [255, 192, 203],
            "coral": [255, 127, 80],
            "salmon": [250, 128, 114],
            "peach": [255, 229, 180],
            "turquoise": [64, 224, 208],
            "teal": [0, 128, 128],
            "cyan": [0, 255, 255],
            "orange": [255, 165, 0],
            "apricot": [251, 206, 177],
        }
        
        # Language-specific color terms mappings
        language_color_mappings = {
            "english": {
                "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
                "purple": "purple", "brown": "brown", "black": "black", "white": "white",
                "gray": "gray", "grey": "gray", "pink": "pink", "orange": "orange",
                "gold": "gold", "silver": "silver", "azure": "azure", "crimson": "crimson",
                "maroon": "maroon", "navy": "navy", "olive": "olive", "violet": "violet",
                "indigo": "indigo", "teal": "teal", "ochre": "ochre", "vermilion": "vermilion"
            },
            
            "latin": {
                "ruber": "red", "rubra": "red", "rubrum": "red",
                "purpureus": "purple", "purpurea": "purple", "purpureum": "purple",
                "caeruleus": "blue", "caerulea": "blue", "caeruleum": "blue",
                "viridis": "green", "viride": "green",
                "flavus": "yellow", "flava": "yellow", "flavum": "yellow",
                "croceus": "orange", "crocea": "orange", "croceum": "orange",
                "luteus": "yellow", "lutea": "yellow", "luteum": "yellow",
                "albus": "white", "alba": "white", "album": "white",
                "candidus": "white", "candida": "white", "candidum": "white",
                "ater": "black", "atra": "black", "atrum": "black",
                "niger": "black", "nigra": "black", "nigrum": "black",
                "aureus": "gold", "aurea": "gold", "aureum": "gold",
                "argenteus": "silver", "argentea": "silver", "argenteum": "silver",
                "ferrugineus": "brown", "ferruginea": "brown", "ferrugineum": "brown",
                "fulvus": "tan", "fulva": "tan", "fulvum": "tan",
                "gilvus": "yellow", "gilva": "yellow", "gilvum": "yellow",
                "glaucus": "blue", "glauca": "blue", "glaucum": "blue",
                "lividus": "blue", "livida": "blue", "lividum": "blue",
                "puniceus": "red", "punicea": "red", "puniceum": "red",
                "roseus": "pink", "rosea": "pink", "roseum": "pink",
                "sanguineus": "red", "sanguinea": "red", "sanguineum": "red",
                "spadix": "brown",
                "vitreus": "cyan", "vitrea": "cyan", "vitreum": "cyan"
            },
            
            "french": {
                "rouge": "red", "bleu": "blue", "vert": "green", "jaune": "yellow",
                "violet": "violet", "marron": "brown", "noir": "black", "blanc": "white",
                "gris": "gray", "rose": "pink", "orange": "orange", "or": "gold",
                "argent": "silver", "turquoise": "turquoise", "cyan": "cyan", 
                "magenta": "magenta", "indigo": "indigo", "marine": "navy", 
                "olive": "olive", "cramoisi": "crimson", "azur": "azure", 
                "beige": "tan", "ocre": "ochre", "pourpre": "purple", 
                "bordeaux": "burgundy", "vermillon": "vermilion"
            },
            
            "italian": {
                "rosso": "red", "blu": "blue", "verde": "green", "giallo": "yellow",
                "viola": "violet", "marrone": "brown", "nero": "black", "bianco": "white",
                "grigio": "gray", "rosa": "pink", "arancione": "orange", "oro": "gold",
                "argento": "silver", "turchese": "turquoise", "ciano": "cyan", 
                "magenta": "magenta", "indaco": "indigo", "marina": "navy", 
                "oliva": "olive", "cremisi": "crimson", "azzurro": "azure", 
                "beige": "tan", "ocra": "ochre", "porpora": "purple", 
                "bordeaux": "burgundy", "vermiglio": "vermilion"
            },
            
            "spanish": {
                "rojo": "red", "azul": "blue", "verde": "green", "amarillo": "yellow",
                "morado": "purple", "marrón": "brown", "negro": "black", "blanco": "white",
                "gris": "gray", "rosa": "pink", "naranja": "orange", "oro": "gold",
                "plata": "silver", "turquesa": "turquoise", "cian": "cyan", 
                "magenta": "magenta", "índigo": "indigo", "marino": "navy", 
                "oliva": "olive", "carmesí": "crimson", "celeste": "azure", 
                "beige": "tan", "ocre": "ochre", "púrpura": "purple", 
                "burdeos": "burgundy", "bermellón": "vermilion"
            },
            
            "german": {
                "rot": "red", "blau": "blue", "grün": "green", "gelb": "yellow",
                "lila": "purple", "braun": "brown", "schwarz": "black", "weiß": "white",
                "grau": "gray", "pink": "pink", "orange": "orange", "gold": "gold",
                "silber": "silver", "türkis": "turquoise", "cyan": "cyan", 
                "magenta": "magenta", "indigo": "indigo", "marine": "navy", 
                "oliv": "olive", "karminrot": "crimson", "azurblau": "azure", 
                "beige": "tan", "ocker": "ochre", "purpur": "purple", 
                "burgunder": "burgundy", "zinnoberrot": "vermilion"
            }
        }
        
        # Build the color database for the current language
        color_db = {}
        language_mapping = language_color_mappings[self.language]
        
        for color_term, base_color in language_mapping.items():
            if base_color in colors_rgb:
                color_db[color_term] = {
                    "rgb": np.array(colors_rgb[base_color]) / 255.0,
                    "embedding": self._get_text_embedding(color_term)
                }
        
        return color_db
    
    def _get_text_embedding(self, text):
        """Get embeddings for a text description"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def _extract_color_descriptions(self, text):
        """Extract color descriptions and their associated regions from text"""
        color_descriptions = []
        
        for pattern in self.color_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    color_term = match[0].lower() if match[0].lower() in self.color_database else match[1].lower()
                    region_term = match[1].strip() if match[0].lower() in self.color_database else match[0].strip()
                    
                    if color_term in self.color_database:
                        color_descriptions.append({"color": color_term, "region": region_term})
        
        return color_descriptions
    
    def _translate_to_english(self, text):
        """Translate non-English text to English for CLIP compatibility"""
        if self.translator is None:
            return text
        
        # Translate in chunks to handle longer texts
        max_length = 512
        if len(text) <= max_length:
            return self.translator(text)[0]['translation_text']
        
        # Split text into chunks
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        translated_chunks = [self.translator(chunk)[0]['translation_text'] for chunk in chunks]
        
        return ' '.join(translated_chunks)
    
    def _find_semantic_regions(self, image, text_descriptions):
        """Segment the image based on semantic descriptions"""
        # Convert image to RGB if it's not
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Initial segmentation using K-means
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Determine optimal K using text descriptions
        k = max(len(text_descriptions) + 2, 5)  # At least 5 clusters or more based on descriptions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Reshape labels and create segmentation mask
        labels = labels.reshape(image.shape[:2])
        
        # Create region proposals
        region_proposals = []
        for i in range(k):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[labels == i] = 255
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out small regions
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    # Skip if the region is too small
                    if w < 20 or h < 20:
                        continue
                    
                    region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(region_mask, [contour], 0, 255, -1)
                    
                    # Extract region image
                    region_img = np.zeros_like(image)
                    region_img[region_mask == 255] = image[region_mask == 255]
                    
                    region_proposals.append({
                        "contour": contour,
                        "mask": region_mask,
                        "image": region_img,
                        "bbox": (x, y, w, h)
                    })
        
        return region_proposals
    
    def _match_regions_to_descriptions(self, image, region_proposals, text_descriptions):
        """Match segmented regions to text descriptions using CLIP"""
        matched_regions = []
        
        # Skip if no regions or descriptions
        if not region_proposals or not text_descriptions:
            return matched_regions
        
        # Prepare image features for each region
        region_images = []
        for region in region_proposals:
            x, y, w, h = region["bbox"]
            # Get region image
            region_img = image[y:y+h, x:x+w]
            # Skip if region is empty
            if region_img.size == 0:
                continue
            # Convert to PIL Image
            region_pil = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
            region_images.append(region_pil)
        
        if not region_images:
            return matched_regions
        
        # Prepare text features for each description - translate if needed
        if self.language != "english":
            text_prompts = [f"a {self._translate_to_english(desc['region'])}" for desc in text_descriptions]
        else:
            text_prompts = [f"a {desc['region']}" for desc in text_descriptions]
        
        # Process images and text with CLIP
        with torch.no_grad():
            # Process in batches if necessary
            batch_size = 16
            similarity_matrix = []
            
            for i in range(0, len(region_images), batch_size):
                batch_images = region_images[i:i+batch_size]
                inputs = self.clip_processor(
                    text=text_prompts, 
                    images=batch_images, 
                    return_tensors="pt", 
                    padding=True
                )
                
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity_matrix.append(logits_per_image)
            
            if similarity_matrix:
                similarity_matrix = torch.cat(similarity_matrix, dim=0)
                # Get the highest similarity match for each region
                region_to_desc_idx = similarity_matrix.argmax(dim=1).cpu().numpy()
                
                # Match regions to descriptions
                for i, desc_idx in enumerate(region_to_desc_idx):
                    if i < len(region_proposals):
                        matched_regions.append({
                            "region": region_proposals[i],
                            "description": text_descriptions[desc_idx],
                            "similarity": similarity_matrix[i, desc_idx].item()
                        })
        
        # Sort by similarity score
        matched_regions.sort(key=lambda x: x["similarity"], reverse=True)
        
        return matched_regions
    
    def _extract_color_from_text(self, color_text):
        """Extract RGB values from color text description"""
        if color_text in self.color_database:
            return self.color_database[color_text]["rgb"]
        
        # Find closest color using embeddings
        color_embedding = self._get_text_embedding(color_text)
        
        best_match = None
        best_score = -float('inf')
        
        for name, data in self.color_database.items():
            similarity = np.dot(color_embedding, data["embedding"]) / (
                np.linalg.norm(color_embedding) * np.linalg.norm(data["embedding"])
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match = name
        
        if best_score > 0.7 and best_match:  # Threshold for confidence
            return self.color_database[best_match]["rgb"]
        else:
            # Default to grayscale if no good match
            return np.array([0.5, 0.5, 0.5])
    
    def _apply_color_to_region(self, image, region_mask, color_rgb):
        """Apply color to a region while preserving texture and details"""
        # Convert to HSV for better color manipulation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Create a color overlay
        color_overlay = np.zeros_like(image, dtype=np.float32)
        color_overlay[:] = color_rgb[::-1] * 255  # BGR format
        color_overlay_hsv = cv2.cvtColor(color_overlay.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Only replace hue and saturation, preserve value (brightness) to maintain texture
        hsv_image_new = hsv_image.copy()
        
        # Apply mask
        mask_3d = np.stack([region_mask] * 3, axis=2) / 255.0
        
        # Transfer hue and saturation, preserve value
        hsv_image_new[:, :, 0] = color_overlay_hsv[:, :, 0] * mask_3d[:, :, 0] + hsv_image[:, :, 0] * (1 - mask_3d[:, :, 0])
        hsv_image_new[:, :, 1] = color_overlay_hsv[:, :, 1] * mask_3d[:, :, 0] + hsv_image[:, :, 1] * (1 - mask_3d[:, :, 0])
        
        # Convert back to BGR
        result_image = cv2.cvtColor(hsv_image_new.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result_image
    
    def process_artwork(self, image, text_description):
        """
        Main method to process artwork and apply color reconstruction
        
        Parameters:
        image: numpy array (BGR format) or path to image file
        text_description: string containing color descriptions
        
        Returns:
        reconstructed_image: numpy array with reconstructed colors
        visualization: dict containing visualization data
        """
        # Load image if string path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError("Could not load image from provided path")
        
        # Make a copy of the original image
        original_image = image.copy()
        working_image = image.copy()
        
        # Extract color descriptions from text
        color_descriptions = self._extract_color_descriptions(text_description)
        print(f"Extracted {len(color_descriptions)} color descriptions in {self.language}:")
        for desc in color_descriptions:
            print(f"  - {desc['region']} -> {desc['color']}")
        
        # Find semantic regions in the image
        region_proposals = self._find_semantic_regions(working_image, color_descriptions)
        print(f"Found {len(region_proposals)} region proposals")
        
        # Match regions to descriptions
        matched_regions = self._match_regions_to_descriptions(working_image, region_proposals, color_descriptions)
        print(f"Matched {len(matched_regions)} regions to descriptions")
        
        # Apply colors to regions
        result_image = working_image.copy()
        visualization_data = {
            "original": original_image,
            "result": None,
            "region_masks": [],
            "matched_descriptions": []
        }
        
        for match in matched_regions:
            region = match["region"]
            description = match["description"]
            
            # Extract color from description
            color_rgb = self._extract_color_from_text(description["color"])
            
            # Apply color to region
            result_image = self._apply_color_to_region(
                result_image, 
                region["mask"], 
                color_rgb
            )
            
            # Save visualization data
            visualization_data["region_masks"].append(region["mask"])
            visualization_data["matched_descriptions"].append(
                f"{description['region']}: {description['color']}"
            )
        
        visualization_data["result"] = result_image
        
        return result_image, visualization_data
    
    def visualize_results(self, visualization_data):
        """Visualize the original, segmentation, and reconstructed image"""
        original = visualization_data["original"]
        result = visualization_data["result"]
        region_masks = visualization_data["region_masks"]
        descriptions = visualization_data["matched_descriptions"]
        
        # Create a combined mask for visualization
        combined_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        for i, mask in enumerate(region_masks):
            # Create colored mask for visualization
            color_mask = np.zeros_like(original)
            color = np.random.randint(0, 256, size=3)
            color_mask[mask > 0] = color
            
            # Add to combined mask with alpha blending
            alpha = 0.4
            combined_mask_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            combined_mask_color = cv2.addWeighted(combined_mask_color, 1, color_mask, alpha, 0)
            combined_mask = cv2.cvtColor(combined_mask_color, cv2.COLOR_BGR2GRAY)
        
        # Create visualization figure
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Segmentation visualization
        plt.subplot(1, 3, 2)
        segmentation_vis = cv2.addWeighted(
            original, 0.7, 
            cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), 0.3, 
            0
        )
        plt.imshow(cv2.cvtColor(segmentation_vis, cv2.COLOR_BGR2RGB))
        plt.title("Segmentation")
        plt.axis('off')
        
        # Reconstructed image
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Reconstructed Image")
        plt.axis('off')
        
        # Add text descriptions
        plt.figtext(0.5, 0.05, "Color descriptions: " + ", ".join(descriptions[:3]) + 
                   ("..." if len(descriptions) > 3 else ""), 
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":5})
        
        plt.tight_layout()
        plt.show()

# Example usage function
def reconstruct_artwork_colors(image_path, text_description):
    """
    Function to demonstrate the usage of the ColorReconstructionAgent
    
    Parameters:
    image_path: Path to the image file
    text_description: Text description containing color information
    
    Returns:
    Reconstructed image and visualization
    """
    agent = MultilingualColorReconstructionAgent()
    result_image, visualization = agent.process_artwork(image_path, text_description)
    agent.visualize_results(visualization)
    return result_image, visualization

# Example usage
if __name__ == "__main__":

    image_path = "faded_fresco.jpg"
    text_description = """
    The ancient fresco shows a scene with red robed figures standing near blue waters.
    There are golden crowns on their heads and the sky appears azure with white clouds.
    The trees have dark green leaves and brown trunks.
    The ground is covered with ochre colored stones and some vermilion flowers.
    """

    result, vis = reconstruct_artwork_colors
