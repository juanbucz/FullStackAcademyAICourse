""" Essentials and Applications of Generative AI_ Unit End Projects - A - Chatbot  

This Unit 6 End Project is STRONGLY based on the following demo from George
Most of the inplace comments still apply.
Any changes/derivations from the base demo are commented

RAG Knowledge System demo

This demo shows how to build a Retrieval-Augmented Generation (RAG) pipeline:
1. **Ingest** - load documents from a source, embed them, store in ChromaDB
2. **Query** - retrieve relevant chunks and pass them as context to an LLM

Architecture:
    Source → Ingestor → Embeddings → ChromaDB
                                        ↓
                          Question → Retriever → Context → Prompt → LLM → Answer

Usage:
    python demos/rag_system/rag_demo.py

Environment variables:
    None required with switch to using ChromaDB 

    To start up ollama client locally:

        # For Image Classification
        $ ollama serve
        $ ollama pull qwen2.5:3b

        # For Image Label Classification
        $ ollama pull llama3    
        $ ollama pull mistral-small:22b

        ** Best Choice
        ** Provides best instruction-following and extraction logic specifically for messy OCR data
        $ ollama pull qwen2.5:14b       
"""

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["FLAGS_use_mkldnn"] = "false"   
os.environ['TF_USE_LEGACY_KERAS'] = '1'    # must be first


import sys
import cv2
import logging
import time
import re 
import inspect
from typing import Tuple, Optional

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

from utilities.utilities import utilities as utils
from utilities.spoonacular_utilities import SpoonacularUtilities as su
from utilities.ocr_ingredient_label_utilities import OCRIngredientLabelUtilities
from utilities.yolov8_utilities import YoloV8Utilities
from utilities.spoonacular_ingredient_mapper_utilities import SpoonacularIngredientMapperUtilities


import requests
import json

from itertools import product
from collections import Counter


import gradio as gr
from dotenv import load_dotenv


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Add src directory to path so relative ingestor imports work
sys.path.insert(0, str(Path(__file__).parent))

#from utilities import PantryUtilities

load_dotenv()

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
__DEFAULT_LOADING_INGREDIENTS = ("assets/loading.png", "Waiting to Load Ingredients...")
__DEFAULT_LOADING_RECIPES     = ("assets/loading.png", "Waiting to Load Recipes...")
__DEFAULT_INGREDIENT_IMAGE    = 'mystery_cloche.jpg'
__UNKNOWN_INGREDIENT_IMAGE    = 'unknown_ingredient.jpg'

__DEFAULT_IMAGES_FOLDER       = 'default_images'
__INGREDIENTS_IMAGES_FOLDER   = 'ingredients_images'
__RECIPES_SCALED_IMAGE_DIR    = 'recipes_scaled_images'

__DEFAULT_IMAGE_PATH          = f'{__DEFAULT_IMAGES_FOLDER}/{__DEFAULT_INGREDIENT_IMAGE}'
__UNKNOWN_INGREDIENT_PATH     = f'{__DEFAULT_IMAGES_FOLDER}/{__UNKNOWN_INGREDIENT_IMAGE}'

MEAT_LIST = ['Whole Chicken', 'Chicken Breast', 'Chicken Thighs', 'Chicken Wings', 'Whole Turkey', 
             'Turkey Breast', 'Ground Turkey', 'Beef Steak', 'Beef Roast', 'Beef Stew Meat', 
             'Ground Beef', 'Pork Chops', 'Pork Tenderloin', 'Pork Shoulder', 'Bacon', 'Ground Pork', 
             'Lamb Chops', 'Ground Lamb', 'Venison']

SEAFOOD_LIST = ['Salmon', 'Tuna', 'Canned Tuna', 'Cod', 
                'Tilapia', 'Halibut', 'Trout', 'Snapper', 'Sea Bass', 'Blue Fish', 
                'Mackerel', 'Sardines', 'Anchovies', 'Swordfish']

SHELLFISH_LIST = ['Shrimp', 'Whole Lobster', 'Lobster Tail', 'Crab Legs', 'Whole Crab', 'Soft Shell Crab', 
                  'Clams', 'Mussels', 'Oysters', 'Scallops', 'Calamari', 'Squid', 'Octopus', 'Crawfish']

CHEESE_LIST = ['Cheddar', 'Mozzarella', 'Parmesan', 'Provolone', 'Swiss', 'Monterey Jack', 'Feta', 
               'Goat Cheese', 'Blue Cheese', 'Brie', 'Gouda', 'Gruyère', 'Ricotta', 'Cream Cheese', 
               'Cottage Cheese', 'Camembert', 'Havarti', 'Pepper Jack', 'Muenster', 'Provolone']

PASTA_LIST = ['Angel Hair', 'Bow Tie', 'Cannelloni', 'Cavatappi', 'Egg Noodles', 'Elbow Macaroni', 'Farfalle',
              'Fettuccine', 'Fusilli', 'Gnocchi', 'Lasagna Sheets', 'Linguine', 'Manicotti', 'Orzo', 'Pappardelle', 
              'Penne', 'Ravioli', 'Rigatoni', 'Rotini', 'Spaghetti', 'Tortellini', 'Ziti']

SPICE_LIST = ['Allspice', 'Anise', 'Cardamom', 'Cayenne Pepper', 'Chili Powder', 'Cinnamon', 'Cloves', 
              'Coriander', 'Cumin', 'Curry Powder', 'Fennel Seed', 'Garlic Powder', 'Ginger', 
              'Mustard Powder', 'Nutmeg', 'Onion Powder', 'Paprika', 'Red Pepper Flakes', 'Smoked Paprika', 
              'Star Anise', 'Sumac', 'Turmeric']

OIL_LIST = ['Avocado Oil', 'Canola Oil', 'Coconut Oil', 'Corn Oil', 'Extra Virgin Olive Oil', 'Grapeseed Oil', 
            'Peanut Oil', 'Safflower Oil', 'Sesame Oil', 'Sunflower Oil', 'Vegetable Oil', 'Walnut Oil']


# ────────────────────────────────────────────────────────────────────────────
# Create Default Gallery Image once instead of every time from URL
# ────────────────────────────────────────────────────────────────────────────
def create_placeholder():

    # Create a 150x150 gray square
    img = Image.new('RGB', (150, 150), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,70), "Loading...", fill=(255,255,0))
    
    # Save to assets folder
    os.makedirs("assets", exist_ok=True)
    img.save('assets/loading.png')

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
__loaded_ingredients = []
__current_ingredients = []
__current_recipes = []


# ────────────────────────────────────────────────────────────────────────────
# Keep track of Enable OCR Scanning Flag
# ────────────────────────────────────────────────────────────────────────────
__use_ocr_image_scanning = False


# ────────────────────────────────────────────────────────────────────────────
# Spoonacular Ingredients Page Data Sources
# ────────────────────────────────────────────────────────────────────────────
create_placeholder()
__ingredient_gallery_items =[__DEFAULT_LOADING_INGREDIENTS]
__current_ingredient_gallery_items =[__DEFAULT_LOADING_INGREDIENTS]

__recipe_gallery_items =[__DEFAULT_LOADING_RECIPES]

__ingredients_id_map = {}
__recipes_id_map = {}

# ────────────────────────────────────────────────────────────────────────────
# Classified and Custom Ingredients Page Data Sources
# ────────────────────────────────────────────────────────────────────────────
__custom_ingredient_gallery_items = [__DEFAULT_LOADING_INGREDIENTS]
__llm_spoonacular_mapping_results = []

# ────────────────────────────────────────────────────────────────────────────
# Configuration - instantiate the class (Loads model to VRAM ONCE)
# ────────────────────────────────────────────────────────────────────────────
__yolo_ingredient_classifier = YoloV8Utilities()
__ocr_label_classifier = OCRIngredientLabelUtilities()
__spoon_mapper_utils = SpoonacularIngredientMapperUtilities()


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────

def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _format_sources(docs) -> str:

    sources = []

    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "")
        preview = doc.page_content[:200].replace("\n", " ")
        sources.append(f"[{i}] {title}\n    {source}\n    \"{preview}...\"")

    return "\n\n".join(sources)


# ────────────────────────────────────────────────────────────────────────────
# Core functions
# ────────────────────────────────────────────────────────────────────────────

def load_ingredients() -> str:
    """Call the Spoonacular API to load the ingredients."""
    global __loaded_ingredients, __ingredient_gallery_items, __ingredients_id_map

    results = su.load_Spoonacular_ingredients()

    # Load ingredients for display/selection
    __loaded_ingredients = su.all_ingredients

    # For 'List Box'
    __ingredient_gallery_items = [(f"{__INGREDIENTS_IMAGES_FOLDER}/{item['image']}", item['name']) for item in __loaded_ingredients]

    # For retrieving ingredient ID
    __ingredients_id_map = {index: item['id'] for index, item in enumerate(__loaded_ingredients)}

    return results, __ingredient_gallery_items

def download_ingredient_images() -> str:        
    """Download the ingredient images from Spoonacular"""
    global __ingredient_gallery_items

    results = su.download_ingredient_images(__ingredient_gallery_items)
    return results, __ingredient_gallery_items

def reset_selected_ingredients() -> tuple[str, any]:
    """Reset/Clear the selected Spoonacular Ingredients."""
    global __current_ingredients, __current_ingredient_gallery_items

    __current_ingredients = []
    __current_ingredient_gallery_items =[__DEFAULT_LOADING_INGREDIENTS]

    return __current_ingredient_gallery_items, __current_ingredient_gallery_items

def clear_ingredients() -> tuple[str, any]:
    """Delete all the loaded Spoonacular Ingredients."""
    print()


def get_recipes() -> str:
    """Call the Spoonacular API to load the recipes."""
    global __current_ingredients, __current_recipes, __recipe_gallery_items, __recipes_id_map

    results = su.load_Spoonacular_recipes(__current_ingredients)

    # Load ingredients for display/selection
    __current_recipes = su.current_recipes

    # For 'List Box'
    __recipe_gallery_items = [(f'{__RECIPES_SCALED_IMAGE_DIR}/{item['title'].replace(" ", "")}.jpg', item['title']) for item in __current_recipes] 

    # For retrieving ingredient ID
    __recipes_id_map = {index: item['id'] for index, item in enumerate(__current_recipes)}       

    return results 

def download_recipe_images() -> str:        
    """Download the recipe images from Spoonacular"""
    global __recipe_gallery_items

    results = su.download_recipe_images()
    return results, __recipe_gallery_items


def parse_recipe_details(json_data):
    """
    Parses Spoonacular recipe detail JSON into a format suitable for Gradio components.
    Includes logic to resolve local image paths and returns HTML for the header.
    """
    if not json_data:
        # Returns placeholders for all 13 outputs defined in the UI
        # Note: The first element is now HTML rather than Markdown
        return ["<h2>No Recipe Selected</h2>"] + ["N/A"] * 12

    # 1. Resolve Local Image and Construct HTML Header
    raw_title = json_data.get('title', 'Unknown Recipe')
    
    # Sanitization logic: removes non-alphanumeric (except . and -) and spaces/ampersands
    clean_name = re.sub(r'[^\w\.-]', '', raw_title.replace(" ", "").replace("&", ""))
    img_type = json_data.get('imageType', 'jpg')
    
    # Path to the ORIGINAL non-scaled image
    local_img_path = f"recipes_images/{clean_name}.{img_type}"
    
    # CRITICAL: Convert to Absolute Path for Gradio's 'file/' protocol
    abs_path = os.path.abspath(local_img_path)

    # Debug check: This will show in your terminal
    #file_exists = os.path.exists(abs_path)    
    # if file_exists:
    #     print(f"Image found for UI: {abs_path}")
    # else:
    #     print(f"Image missing for UI: {abs_path}")

    # Use gr.HTML instead of Markdown to prevent protocol stripping
    # header_html = f"""
    # <div style="text-align: center; font-family: sans-serif;">
    #     <h1 style="margin-bottom: 10px;">{raw_title}</h1>
    #     <img src="file/{abs_path}" 
    #          width="100%" 
    #          style="border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-height: 400px; object-fit: cover;"
    #          alt="{raw_title}">
    # </div>
    # """
    image_url = json_data.get('image', '')
    
    header_markdown = f'<h2>📜 {raw_title}</h2>'

    # 2. Extract Metadata
    source = json_data.get('sourceName', 'N/A')
    score = round(json_data.get('spoonacularScore', 0), 2)
    price = round(json_data.get('pricePerServing', 0) / 100, 2) # Cents to Dollars

    # 3. Dish Types & Summary
    dish_list = json_data.get('dishTypes', [])
    if dish_list:
        dish_types = ", ".join([dt.replace("_", " ").title() for dt in dish_list])
    else:
        dish_types = "None specified"

    summary = json_data.get('summary', 'No summary available.')

    # 4. Ingredients & Instructions
    ing_list = json_data.get('extendedIngredients', [])
    ingredients_text = "\n".join([f"• {i.get('original')}" for i in ing_list])
    instructions = json_data.get('instructions', "No instructions provided.")

    # 5. Wine, Cuisines, and Diets
    # Access the winePairing dictionary safely
    wine_pairing_data = json_data.get('winePairing', {})

    # Extract only the list of specific wine names
    wine_list = wine_pairing_data.get('pairedWines', [])

    # Format the list into a clean, capitalized string
    if wine_list:
        # This uses .title() to turn 'chardonnay' into 'Chardonnay'
        wine_text = ", ".join([wine.title() for wine in wine_list])
    else:
        wine_text = "None suggested"    
    cuisines_text = ", ".join(json_data.get('cuisines', []))
    diets_text = ", ".join(json_data.get('diets', []))

    # 6. Taste Profile & Nutrition
    # Scale values by 100 so '100' becomes '1.0' (100%)
    taste_raw = json_data.get('taste', {})
    taste_data = {k: v / 100 for k, v in taste_raw.items()}
    
    nutrients = json_data.get('nutrition', {}).get('nutrients', [])
    key_macros = [n for n in nutrients if n['name'] in ['Calories', 'Fat', 'Carbohydrates', 'Protein']]
    nutrition_text = "\n".join([
        f"{m['name']}: {m['amount']}{m['unit']} ({m['percentOfDailyNeeds']}% of daily need)" 
        for m in key_macros
    ])

    return (
        header_markdown, source, score, price, dish_types, summary, 
        ingredients_text, abs_path,instructions, wine_text, 
        cuisines_text, taste_data, diets_text, nutrition_text
    )

def load_recipe_details(recipe_id, recipe_name) -> tuple[str, any]:
    """Load recipe details for the selected Spoonacular recipe."""
    
    results = su.load_recipe_details(recipe_id, recipe_name)
    recipe_details = su.current_recipe_details
    return parse_recipe_details(recipe_details)

def clear_recipes_ui():
    """Resets the entire Recipe Management tab state and UI components."""

    # 1. Reset Global State
    global __current_recipes, __recipe_gallery_items, __recipes_id_map
    __current_recipes = []
    __recipe_gallery_items = [__DEFAULT_LOADING_RECIPES]
    __recipes_id_map = {}

    # 2. Reset Galleries & Status
    gallery_reset = __DEFAULT_LOADING_RECIPES # For the recipe list
    status_text = "Recipes cleared. Please select ingredients and load again."
    
    # 3. Reset Recipe Selection Summary (Center Panel)
    summary_reset = ["", 0, 0, "", 0, "", None, None] # title, likes, missing count/list, unused count/list, state IDs
    
    # 4. Reset Recipe Detail Panel (Right Panel)
    detail_title_reset = "<h2>📜 Select a Recipe to See Details</h2>"
    detail_meta_reset = ["", 0, 0, "", ""] # source, score, price, dish types, summary
    detail_body_reset = ["", None, ""]     # ingredients, image, instructions
    detail_footer_reset = ["", "", {}, "", ""] # wine, cuisines, taste (dict), diets, nutrition
    
    # Combine all into one return list
    return (
        [gallery_reset, status_text] + 
        summary_reset + 
        [detail_title_reset] + 
        detail_meta_reset + 
        detail_body_reset + 
        detail_footer_reset
    )

# ────────────────────────────────────────────────────────────────────────────
# Image Classifier Functions
# ────────────────────────────────────────────────────────────────────────────

def update_ocr_scanning_flag(use_ocr_scanning):
    """ """
    global __use_ocr_image_scanning

    __use_ocr_image_scanning = use_ocr_scanning


def classify_ingredient(image) -> Tuple[Optional[str], Optional[str]]:
    
    # Check to see if OCR label scanning is enabled
    print(f'OCR Label SCANNING ENABLED: {__use_ocr_image_scanning}')
    ingredient_name = None

    # If OCR scanning is enabled, attempt to ID image via ingredient label first
    if __use_ocr_image_scanning == True:
        ingredient_name = __ocr_label_classifier.identify_ingredient(image)
        conf = .95

    # Predict ingredient label/confidence form yolo util library
    if ingredient_name is None or 'unknown' in ingredient_name.lower():
        ingredient_name, conf = __yolo_ingredient_classifier.classify_ingredient(image)

    confidence = f"{conf:.2f}"

    return ingredient_name, confidence

def add_classified_ingredient(ingredient_name, image_array):
    """ """
    global __current_ingredient_gallery_items

    # Check if ingredient name exists before adding
    if any(item[1].lower() == ingredient_name.lower() for item in __custom_ingredient_gallery_items):
        # Return the gallery, the original image, the original name, and the original confidence
        # gr.update() - don't change anything in the associated output control
        return (
            __current_ingredient_gallery_items, 
            gr.update(), 
            gr.update(), 
            gr.update()
        )
    
    # Remove default list place holder __DEFAULT_LOADING_INGREDIENTS
    if __DEFAULT_LOADING_INGREDIENTS in __current_ingredient_gallery_items:
        __current_ingredient_gallery_items.remove(__DEFAULT_LOADING_INGREDIENTS)    

    # Save a jpeg of the image with all the images on disk
    image_path = f'{__INGREDIENTS_IMAGES_FOLDER}/classified_{ingredient_name}.jpg'

    # Convert RGB (Gradio default) to BGR (OpenCV default)
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Save the file
    cv2.imwrite(image_path, bgr_image)

    # Add to the list (alphabetical order)
    __current_ingredient_gallery_items.append((image_path, ingredient_name))

    # Sort alphabetically
    __current_ingredient_gallery_items.sort(key=lambda x: x[1])    

    return __current_ingredient_gallery_items, None, '', ''

def add_standard_ingredients(meat, fish, shellfish, pasta, spices):
    """ """
    global __current_ingredient_gallery_items

    # Remove default list place holder __DEFAULT_LOADING_INGREDIENTS
    if __DEFAULT_LOADING_INGREDIENTS in __current_ingredient_gallery_items:
        __current_ingredient_gallery_items.remove(__DEFAULT_LOADING_INGREDIENTS)

    new_ingredients = []

    if meat is not None:
        new_ingredients.append(meat)

    if fish is not None:
        new_ingredients.append(fish)

    if isinstance(shellfish, list) and shellfish:
        new_ingredients.extend(shellfish)
    if pasta is not None:
        new_ingredients.append(pasta)

    if isinstance(spices, list) and spices:
        new_ingredients.extend(spices)

    # Remove any pre-exiting items from the list
    existing_names = {item[1] for item in __current_ingredient_gallery_items}
    items_to_add = [name for name in new_ingredients if name not in existing_names]

    __current_ingredient_gallery_items.extend([(__DEFAULT_IMAGE_PATH, item) for item in items_to_add]) 

    # Sort alphabetically
    __current_ingredient_gallery_items.sort(key=lambda x: x[1])

    return __current_ingredient_gallery_items, None, None, [], None, []

# ────────────────────────────────────────────────────────────────────────────
# Map ingredient names to Spoonacular vocabulary
# ────────────────────────────────────────────────────────────────────────────
    
def translate_ingredient_names_to_spoonacular():
    """ """
    global __custom_ingredient_gallery_items, __llm_spoonacular_mapping_results

    if not __current_ingredient_gallery_items:
        return __current_ingredient_gallery_items

    # Get the existing names from the gallery
    existing_names = {item[1] for item in __custom_ingredient_gallery_items}

    # Pass names into mapper for name resolution
    ingredient_dict = __spoon_mapper_utils.map_ingredients_to_spoonacular(existing_names)

    # For more efficient resolution/lookups create dictionary based on gallery_items
    # Format: { "label": "current_path" }
    gallery_lookup = {label: path for path, label in __custom_ingredient_gallery_items}
    
    if __llm_spoonacular_mapping_results:
        __llm_spoonacular_mapping_results.clear()

    __llm_spoonacular_mapping_results = [(label, mapped_label) for label, mapped_label in ingredient_dict.items()]

    # By examining the current gallery, figure out which ingredients need downloaded images.
    # If it has "default image_path" it needs to have image downloaded
    download_queue = []
    for original_label, mapped_label in ingredient_dict.items():

        # If this was a recognized ingredients
        if mapped_label != "unknown":
            current_path = gallery_lookup.get(original_label)

            if current_path == __DEFAULT_IMAGE_PATH:
                download_queue.append(mapped_label)

    # Download images for unique ingredient names
    successful_names = set()
    if download_queue:
        unique_requests = list(set(download_queue))
        successful_names = su.download_ingredient_images(unique_requests)

    updated_gallery = []
    for original_label, mapped_label in ingredient_dict.items():
        if original_label in gallery_lookup:
            current_path = gallery_lookup[original_label]
            
            # A: Item is not food
            if mapped_label == "unknown":
                new_path = __UNKNOWN_INGREDIENT_PATH
            
            # B: Item is food AND currently a placeholder
            elif current_path == __DEFAULT_IMAGE_PATH:
                # Check if the download actually landed on disk
                if mapped_label in successful_names:
                    new_path = f"{__INGREDIENTS_IMAGES_FOLDER}/{mapped_label}.jpg"
                else:
                    # Fallback to default if API returned 404/Error
                    new_path = __DEFAULT_IMAGE_PATH
            
            # C: Item already has a custom/scanned image
            else:
                new_path = current_path
            
            updated_gallery.append((new_path, original_label))

    # Sort alphabetically
    __custom_ingredient_gallery_items = updated_gallery    
    __custom_ingredient_gallery_items.sort(key=lambda x: x[1])

    return __custom_ingredient_gallery_items, __llm_spoonacular_mapping_results


# ────────────────────────────────────────────────────────────────────────────
# Load classified/translated ingredients from ingredient management to recipe management
# ────────────────────────────────────────────────────────────────────────────
def load_ingredients_into_recipe_management():
    """ """
    global __current_ingredient_gallery_items, __custom_ingredient_gallery_items, __current_ingredients

    if __current_ingredient_gallery_items:
        __current_ingredient_gallery_items.clear()

    if __current_ingredients:
        __current_ingredients.clear()

    __current_ingredient_gallery_items.extend(__custom_ingredient_gallery_items)
    __current_ingredients = [label for _, label in __custom_ingredient_gallery_items]

    return __current_ingredient_gallery_items


# ────────────────────────────────────────────────────────────────────────────
# Clear/Remove ingredients from Gallery
# ────────────────────────────────────────────────────────────────────────────
def remove_ingredient_from_gallery(data: gr.SelectData):
    """ """
    global __custom_ingredient_gallery_items

    # If default is the only left, return - don't do anything
    if len(__custom_ingredient_gallery_items) == 1 and __custom_ingredient_gallery_items[0] == __DEFAULT_LOADING_INGREDIENTS:
        return __custom_ingredient_gallery_items
    
    # Get the specific item clicked from the master pantry
    clicked_item = __custom_ingredient_gallery_items[data.index] 
    __custom_ingredient_gallery_items.remove(clicked_item)

    # Add default place holder if list empty
    if not __custom_ingredient_gallery_items:
        __custom_ingredient_gallery_items.append(__DEFAULT_LOADING_INGREDIENTS)

    # Sort alphabetically
    __custom_ingredient_gallery_items.sort(key=lambda x: x[1])
    return __custom_ingredient_gallery_items

def clear_all_ingredients():
    """ """
    global __ingredient_gallery_items, __llm_spoonacular_mapping_results
    
    __ingredient_gallery_items.clear()
    __ingredient_gallery_items.append(__DEFAULT_LOADING_INGREDIENTS)

    __custom_ingredient_gallery_items.clear()
    __custom_ingredient_gallery_items.append(__DEFAULT_LOADING_INGREDIENTS)

    if __llm_spoonacular_mapping_results:
        __llm_spoonacular_mapping_results.clear()

    return None, '', '', None, None, [], None, [], '', __ingredient_gallery_items, __llm_spoonacular_mapping_results

# ────────────────────────────────────────────────────────────────────────────
# Add Custom Ingredients
# ────────────────────────────────────────────────────────────────────────────

def add_custom_ingredients(custom_ingredients): 
    """ """   
    global __custom_ingredient_gallery_items

    if not custom_ingredients:
        return __current_ingredient_gallery_items, ''
    
    # Use list comprehension to split, strip, and ignore empty results
    new_ingredients = [item.strip() for item in custom_ingredients.split(",") if item.strip()]   

    # Remove any pre-exiting items from the list
    existing_names = {item[1] for item in __custom_ingredient_gallery_items}
    items_to_add = [name for name in new_ingredients if name not in existing_names]

    __custom_ingredient_gallery_items.extend([(__DEFAULT_IMAGE_PATH, item) for item in items_to_add])      
                    
    return  __custom_ingredient_gallery_items, ''

# ────────────────────────────────────────────────────────────────────────────
# OCR Label Scanning Functions
# ────────────────────────────────────────────────────────────────────────────    

# ───────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
# ───────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Define the CSS (Critical: 'overflow-y' must be 'auto' or 'scroll')
# ---------------------------------------------------------------------------
custom_css = """
#scroll_list { 
    height: 600px; 
    overflow-y: auto !important; 
    border: 1px solid #ddd; 
    padding: 10px;
}
"""
## Hides the entire top-right control bar (Upload, Clear, etc.) 
# This string is passed to gr.Blocks(css=...)
# The final combined CSS for your pantry app
combined_pantry_css = """
/* 1. HIDE GRADIO UI CONTROLS */
.gallery-container .controls, .gallery-container .selected-controls,
button[aria-label="Clear"], button[aria-label="Upload"] {
    display: none !important;
}

/* 2. NEUTRALIZE GRID CONTAINER */
#pantry_list .grid-container {
    display: flex !important;
    flex-direction: column !important;
    gap: 0px !important;
    padding: 0px !important;
    border: none !important;
}

/* 3. STRIP BUTTON WRAPPER (DEFAULT & SELECTED) */
/* 'all: unset' kills the hidden padding/min-height that causes gaps */
#pantry_list .grid-container > button,
#pantry_list .grid-container > button.selected {
    all: unset !important;
    display: block !important;
    height: 60px !important;
    width: 100% !important;
    margin: 0px !important;
    padding: 0px !important;
    cursor: pointer !important;
}

/* 4. ROW LAYOUT (DEFAULT & SELECTED) */
/* We target .selected to ensure clicking doesn't change the layout */
#pantry_list .thumbnail-item,
#pantry_list .thumbnail-item.selected {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    height: 60px !important;
    width: 100% !important;
    padding: 0px 10px !important;
    margin: 0px !important;
    border-bottom: 1px solid #eee !important;
    background: white !important;
}


/* 5. SELECTION FEEDBACK */
/* Adds a subtle color so you know it was clicked without it growing large */
#pantry_list .thumbnail-item.selected {
    background-color: #f0f7ff !important;
}

/* 6. IMAGE SIZE LOCK (DEFAULT & SELECTED) */
#pantry_list .thumbnail-item img,
#pantry_list .thumbnail-item.selected img {
    width: 50px !important;
    height: 50px !important;
    object-fit: contain !important;
    margin-right: 15px !important;
    flex-shrink: 0 !important;
    /* Prevent Gradio from centering/scaling the image when selected */
    position: static !important;
    transform: none !important;
}

/* 7. TEXT STYLE LOCK (DEFAULT & SELECTED) */
#pantry_list .caption-label,
#pantry_list .thumbnail-item.selected .caption-label {
    font-size: 18px !important;
    font-weight: 500 !important;
    color: #333 !important;
    position: static !important;
    background: transparent !important;
    padding: 0px !important;
    margin: 0px !important;
    box-shadow: none !important;
}

/* 8. DISABLE PREVIEW MODE LIGHTBOX */
.preview {
    display: none !important;
}
"""

with gr.Blocks(title="Unit to Pantry Recipe Selection System", css=combined_pantry_css) as pantry_to_table:

    with gr.Tabs():

        
        # ------------------------------------------------------------------
        # Tab 1: Ingredients Upload, Classification and  Management
        # ------------------------------------------------------------------
        with gr.Tab("1. General Instructions"):
            gr.Markdown("""
                # This system uses allows users to gather a list of ingredients and generate recipe possibilities
                
                **How it works:**
                1. **Manage Custom User Ingredients** - Upload, Classify and Manage Images  
                2. **Manage Spoonacular Ingredients** - calls out to Spoonacular API to retrieve requested number ingredients and their matching images
                3. **Manage Recipes** - Based on selected ingredients retrieve a list of recipes
                </br>
                """)
            
            gr.Markdown(inspect.cleandoc("""
                ## Manage Custom User Ingredients
                1. Upload ingredient image file in jpeg format.
                2. Classify image using YoloV8m model</br>
                &nbsp;&nbsp;&nbsp;&nbsp;a. Accept image classification </br>
                &nbsp;&nbsp;&nbsp;&nbsp;b. Reject image classification - keep image but enter desired ingredient name</br>
                &nbsp;&nbsp;&nbsp;&nbsp;c. Reject image classification - clear image and load another one</br>
                3. Map classified ingredient to Spoonacular ingredient name vocabulary
                4. Add Spoonacularized ingredient and image to the ingredient list
                5. When completed, navigate to Manage Recipes tab to generate recipes.
                </br>                                         
                """))        
            
            gr.Markdown("""
                ## Manage Spoonacular Ingredients
                1. Click **Load Spoonacular** Ingredients
                2. Click **Download Spoonacular** Ingredient Images
                3. Select Available Pantry ingredient to *add it* to the Recipe Ingredients
                4. Select Recipe ingredient to *remove it* from Recipe Ingredient list
                5. Click on **Reset Recipe Ingredients** to remove all recipe ingredients
                6. Click on **Clear All Ingredients** to to reset this form.
                7. When completed, navigate to Manage Recipes tab to generate recipes.
                </br>                        
                """)        
            
            gr.Markdown("""
                ## Manage Recipes
                1. Click **Load Spoonacular Recipes** to retrieve recipes based on ingredient list
                2. Click **Download Recipe Images** to see recipe images
                3. Select recipe from Available Recipes to see recipe preview (bottom panel)
                4. Click **Load Selected Recipe** to view recipe details (right panel)
                5. Click **Clear All Recipe Data** to reset this form.
                """)                    

        # ------------------------------------------------------------------
        # Tab 1: Ingredients Upload, Classification and  Management
        # ------------------------------------------------------------------
        with gr.Tab("1. Manage Custom User Ingredients"):
            gr.Markdown("""
                        # Upload, Classify and Manage Ingredients
                        """)
            
            # Left Side column
            with gr.Column(scale=3):

                selected_indices = gr.State([])

                with gr.Row():
                        
                        # Image Loading
                        with gr.Column(scale=2, min_width=300):
                            with gr.Group():
                                gr.Markdown("### 📝 Load Ingredient image")
                                input_image1 = gr.Image(
                                                        label="Upload or Take a Photo of an Ingredient",
                                                        sources=["upload", "webcam"], # Enables both entry methods
                                                        type="numpy"                  # Standard format for YOLO/OpenCV
                                                    )        

                                classified_ingredient_name1 = gr.Textbox(
                                                                    label="Ingredient Name",
                                                                    placeholder="e.g., olive oil, mustard, heavy cream",
                                                                    interactive=False,
                                                                    lines=2
                                                                )
                                confidence_level1 = gr.Textbox(
                                                            label="Classification Confidence Level",
                                                            placeholder="0.0",
                                                            interactive=False,
                                                            lines=2
                                                        )
                                
                                use_ocr_scanning_cb1 = gr.Checkbox(
                                        label="Enable OCR Image Scanning", 
                                        value=__use_ocr_image_scanning,
                                        info='Check this to add ingredient labels scanning.'
                                    )

                                classify_ingredients_btn1 = gr.Button("Classify Ingredient", variant="primary")
                                clear_classified_ingredient_name_btn1 = gr.Button("Clear Classified Ingredient")    
                                add_classified_btn1 = gr.Button("Add Classified Ingredient", variant="primary")


                        # The Additional Ingredients
                        with gr.Column(scale=2, min_width=400):
                            with gr.Group():                            
                                gr.Markdown("### 📝 Select Additional Ingredients",
                                            label='')
                                
                                gr.Textbox(value="Standard Ingredients List", interactive=False)

                                meat_dropdown = gr.Dropdown(choices=MEAT_LIST, 
                                                            value=None,
                                                            label='Meat',
                                                            container=True,
                                                            interactive=True)
                                
                                fish_dropdown = gr.Dropdown(choices=SEAFOOD_LIST,
                                                            value=None,
                                                            label='Fish',
                                                            container=True,
                                                            interactive=True)                            
                                                            
                                shellfish_dropdown = gr.Dropdown(choices=SHELLFISH_LIST,
                                                                value=None,
                                                                multiselect=True,
                                                                label='Shellfish',
                                                                container=True,
                                                                interactive=True)                                                        
                                
                                pasta_dropdown = gr.Dropdown(choices=PASTA_LIST,
                                                            value=None,
                                                            label='Pasta',
                                                            container=True,
                                                            interactive=True)                                                        
                                
                                spice_dropdown = gr.Dropdown(choices=SPICE_LIST,
                                                            value=None,
                                                            multiselect=True,
                                                            label='Spices',
                                                            container=True,
                                                            interactive=True)       
                                      
                                add_standard_ingredients_btn = gr.Button("Add Standard Ingredients", variant="primary")
                                clear_standard_ingredients_btn = gr.Button("Clear Standard Ingredients")    

                                custom_ingredients = gr.Textbox(
                                                            label="Custom Ingredients (comma separated)",
                                                            placeholder="e.g., olive oil, mustard, heavy cream",
                                                            lines=2,
                                                            interactive=True
                                                        )

                                add_custom_ingredients_btn = gr.Button("Add Custom Ingredients", variant="primary")
                                clear_custom_ingredients_btn = gr.Button("Clear Custom Ingredients")    
                                

                         # Ingredients List
                        with gr.Column(scale=3, min_width=400):
                            with gr.Group():
                                gr.Markdown("### 📝 Desired Ingredients")
                                custom_ingredient_recipe_gallery1 = gr.Gallery(
                                    value=__custom_ingredient_gallery_items, 
                                    elem_id="pantry_list", 
                                    columns=1, 
                                    height=600, 
                                    interactive=True
                                )

                                translate_to_spoonacular_btn = gr.Button("Translate to Spoonacular", variant="primary")
                                spoonacular_ingredients_to_recipe_btn = gr.Button("Load Ingredients into Recipe Management", variant="primary")
                                clear_all_ingredients_btn = gr.Button("Clear All Ingredients")  

                                gr.Markdown("### 📝 LLM Spoonacular Mapping Results")
                                llm_results_dataframe = gr.DataFrame(
                                    value=__llm_spoonacular_mapping_results, 
                                    headers=["Raw Label", "Spoonacular Match"],
                                    datatype=["str", "str"]
                                )

                # use_ocr_scanning_cb.change(
                #     fn=update_ocr_scanning_flag, 
                #     inputs=[use_ocr_scanning_cb])                  

                # classify_ingredients_btn.click(
                #     fn=classify_ingredient,
                #     inputs=[input_image],
                #     outputs=[classified_ingredient_name, confidence_level]
                # )

                # clear_classified_ingredient_name_btn.click(
                #     fn=lambda: (None, '', ''),
                #     outputs=[input_image, classified_ingredient_name, confidence_level]
                # )                                                                   

                # add_classified_btn.click(
                #     fn=add_classified_ingredient,
                #     inputs=[classified_ingredient_name, input_image],
                #     outputs=[custom_ingredient_recipe_gallery, input_image, classified_ingredient_name, confidence_level]
                # )                                                              

                # add_standard_ingredients_btn.click(
                #     fn=add_standard_ingredients,
                #     inputs=[meat_dropdown, fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown],
                #     outputs=[custom_ingredient_recipe_gallery, meat_dropdown, fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown]
                # )

                # clear_standard_ingredients_btn.click(
                #     fn=lambda: (gr.update(value=None), 
                #                 gr.update(value=None),
                #                 gr.update(value=[]),
                #                 gr.update(value=None),
                #                 gr.update(value=[])),
                #     outputs=[meat_dropdown, fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown]
                # )                                 
              
                # add_custom_ingredients_btn.click(
                #     fn=add_custom_ingredients,
                #     inputs=[custom_ingredients],
                #     outputs=[custom_ingredient_recipe_gallery, custom_ingredients]                    
                # )

                # clear_custom_ingredients_btn.click(
                #     fn=lambda: (''),
                #     outputs=[custom_ingredients]
                # )                                        

                # translate_to_spoonacular_btn.click(
                #     fn=translate_ingredient_names_to_spoonacular,
                #     outputs=[custom_ingredient_recipe_gallery, llm_results_dataframe]
                # )

                # custom_ingredient_recipe_gallery.select(fn=remove_ingredient_from_gallery, 
                #                        outputs=[custom_ingredient_recipe_gallery]) 

                # clear_all_ingredients_btn.click(
                #     fn=clear_all_ingredients,
                #     outputs=[input_image, classified_ingredient_name, confidence_level, meat_dropdown,
                #              fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown, 
                #              custom_ingredients, custom_ingredient_recipe_gallery, llm_results_dataframe]
                # )                     


        # ------------------------------------------------------------------
        # Tab 3: Spoonacular Ingredient Management
        # ------------------------------------------------------------------
        with gr.Tab("2. Manage Spoonacular Ingredients"):
            gr.Markdown("""
                        # Loads requested number of ingredients, and their related images via Spoonacular API
                        """)

            # State to hold multiple selected IDs
            selected_ingredients_state = gr.State(value=[])

            with gr.Row():
                with gr.Column():

                    with gr.Row():
                        load_ingredients_btn = gr.Button("Load Spoonacular Ingredients", variant="primary")
                        download_ingredient_images_btn = gr.Button("Download Ingredient Images", variant="primary")
                        with gr.Column(min_width=150):
                            reset_ingredients_btn = gr.Button("Reset Recipe Ingredients", variant="stop")
                            clear_ingredients_btn = gr.Button("Clear Spoonacular Ingredients", variant="stop")

                    gr.Markdown("### 🛒 Food Pantry Inventory")

                    selected_indices = gr.State([])

                    with gr.Row():
                            # The Source Gallery (Your 1,000+ items)
                            ingredients_gallery = gr.Gallery(
                                value=__ingredient_gallery_items, 
                                label="Available Pantry", 
                                elem_id="pantry_list", # Uses your verified CSS
                                columns=1, 
                                height=600, 
                                interactive=True
                            )

                            # The Selection Gallery (Items ready for the recipe)
                            recipe_ingredients_gallery = gr.Gallery(
                                value=__current_ingredient_gallery_items, 
                                label="Selected for Recipe", 
                                elem_id="pantry_list", # Re-use the same CSS for consistency
                                columns=1, 
                                height=600, 
                                interactive=True
                            )
                    
                with gr.Column():
                    ingredient_load_status = gr.Textbox(label="Ingredients Status", lines=4, interactive=False)


            load_ingredients_btn.click(
                fn=load_ingredients,
                outputs=[ingredient_load_status, ingredients_gallery],
            )

            download_ingredient_images_btn.click(
                fn=download_ingredient_images,
                outputs=[ingredient_load_status, ingredients_gallery],
            )            

        # ------------------------------------------------------------------
        # Tab 4: Recipe Management
        # ------------------------------------------------------------------
        with gr.Tab("3. Manage Recipes"):
            gr.Markdown("""
            # Loads top ranked recipes and their images based on selected ingredients via Spoonacular API
            """)

            selected_recipe_id = gr.State(value=None)
            selected_recipe_name = gr.State(value=None)

            with gr.Row():

                # Left Side column
                with gr.Column(scale=2):

                    with gr.Row():
                        get_recipes_btn = gr.Button("Load Spoonacular Recipes", variant="primary", visible=False)
                        download_recipe_images_btn = gr.Button("Download Recipe Images", variant="primary", visible=False)
                        with gr.Column(min_width=150):
                            load_recipe_button = gr.Button("Load Selected Recipe", variant="primary", visible=False)
                            clear_recipes_btn = gr.Button("Clear All Recipe Data", variant="stop", visible=False)

                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group():

                                gr.Markdown("### 📝 Load Ingredient image")

                                input_image = gr.Image(
                                                        label="Upload or Take a Photo of an Ingredient",
                                                        sources=["upload", "webcam"], # Enables both entry methods
                                                        type="numpy"                  # Standard format for YOLO/OpenCV
                                                    )        
                                
                                with gr.Row():
                                
                                    classified_ingredient_name = gr.Textbox(
                                                                        label="Ingredient Name",
                                                                        placeholder="e.g., olive oil, mustard, heavy cream",
                                                                        interactive=False,
                                                                        lines=2,
                                                                        scale=7
                                                                    )
                                    confidence_level = gr.Textbox(
                                                                label="Confidence",
                                                                placeholder="0.0",
                                                                interactive=False,
                                                                lines=2,
                                                                scale=1,
                                                                min_width=80
                                                            )
                                
                                use_ocr_scanning_cb = gr.Checkbox(
                                        label="Enable OCR Image Scanning", 
                                        value=__use_ocr_image_scanning,
                                        info='Check this to add ingredient labels scanning.'
                                    )

                                classify_ingredients_btn = gr.Button("Classify Ingredient", variant="primary")
                                add_classified_btn = gr.Button("Add Classified Ingredient", variant="huggingface")                            

                        with gr.Column(scale=2):                            

                            gr.Markdown("### 📝 Select Additional Ingredients")

                            with gr.Row():

                                meat_dropdown = gr.Dropdown(choices=MEAT_LIST, 
                                                            value=None,
                                                            label='Meat',
                                                            container=True,
                                                            interactive=True)
                                fish_dropdown = gr.Dropdown(choices=SEAFOOD_LIST,
                                                            value=None,
                                                            label='Fish',
                                                            container=True,
                                                            interactive=True)                            

                            with gr.Row():

                                shellfish_dropdown = gr.Dropdown(choices=SHELLFISH_LIST,
                                                                value=None,
                                                                multiselect=True,
                                                                label='Shellfish',
                                                                container=True,
                                                                interactive=True)                                                        

                                pasta_dropdown = gr.Dropdown(choices=PASTA_LIST,
                                                            value=None,
                                                            label='Pasta',
                                                            container=True,
                                                            interactive=True)                                                        

                            with gr.Row():
                                                                
                                spice_dropdown = gr.Dropdown(choices=SPICE_LIST,
                                                            value=None,
                                                            multiselect=True,
                                                            label='Spices',
                                                            container=True,
                                                            interactive=True) 
                                
                                oil_dropdown = gr.Dropdown(choices=OIL_LIST,
                                                            value=None,
                                                            multiselect=True,
                                                            label='Oils',
                                                            container=True,
                                                            interactive=True)                                  
                                
                                cheese_dropdown = gr.Dropdown(choices=CHEESE_LIST,
                                                              value=None,
                                                              multiselect=True,
                                                              label='Cheeses',
                                                              container=True,
                                                              interactive=True)  
                                
                            add_standard_ingredients_btn = gr.Button("Add Standard Ingredients", variant="huggingface")

                            custom_ingredients = gr.Textbox(
                                                        label="Custom Ingredients (comma separated)",
                                                        placeholder="e.g., olive oil, mustard, heavy cream",
                                                        lines=2,
                                                        interactive=True
                                                    )

                            add_custom_ingredients_btn = gr.Button("Add Custom Ingredients", variant='huggingface')
                            clear_custom_ingredients_btn = gr.Button("Restart Recipe", variant='stop')    


                    selected_indices = gr.State([])

                    # with gr.Row():
                            
                    #         with gr.Column(scale=2, min_width=300):
                    #             # The Selected Ingredients Gallery (Items ready for the recipe)
                    #             selected_ingredients_gallery = gr.Gallery(
                    #                 value=__current_ingredient_gallery_items, 
                    #                 elem_id='pantry_ingredient_list', 
                    #                 label='Selected Ingredients',
                    #                 columns=3, 
                    #                 height=300, 
                    #                 interactive=True
                    #             )

                    #         # The Available Recipes Gallery 
                    #         with gr.Column(scale=3, min_width=400):
                    #             recipe_gallery = gr.Gallery(
                    #                 value=__recipe_gallery_items, 
                    #                 elem_id='pantry_recipe_list', 
                    #                 label='Available Recipes',
                    #                 columns=3, 
                    #                 height=300, 
                    #                 interactive=True
                    #             )

                    with gr.Row():
                        # Force the Ingredients list to be much narrower
                        with gr.Column(scale=1, min_width=200): # Dropped scale from 2 to 1
                            selected_ingredients_gallery = gr.Gallery(
                                value=__current_ingredient_gallery_items, 
                                elem_id='pantry_ingredient_list', 
                                label='Selected Ingredients',
                                columns=1, # Change back to 1 for a clean vertical list
                                height=300, 
                                object_fit="contain" # CRITICAL: stops the image from stretching
                            )

                        # Give the Recipes the lion's share of the 50% left-side block
                        with gr.Column(scale=2, min_width=300): # Kept at 2 or 3
                            recipe_gallery = gr.Gallery(
                                value=__recipe_gallery_items, 
                                elem_id='pantry_recipe_list',                                 
                                label='Available Recipes',
                                columns=2, # 2 columns here looks better in a split view
                                height=300, 
                                object_fit="contain"
                            )                    
                    
                    gr.Markdown("### 🔍 Selected Recipe Details")
                    with gr.Group(): # Group these related fields together visually
                        with gr.Row():
                            recipe_title_display = gr.Textbox(label="Recipe Title", interactive=False)
                            recipe_likes_display = gr.Number(label="Spoonacular Likes", interactive=False)

                        with gr.Row():
                            with gr.Column():
                                missing_count_display = gr.Number(label="# Missing Ingredients", interactive=False)
                                missing_ingredients_list = gr.Textbox(
                                    label="Missing Items (Shopping List)", 
                                    lines=5,       # Starting height
                                    max_lines=8,   # Maximum height before scrollbar locks in
                                    interactive=False
                                )

                            with gr.Column():
                                unused_count_display = gr.Number(label="# Unused Ingredients", interactive=False)
                                unused_ingredients_list = gr.Textbox(
                                    label="Unused Items (Already in Pantry)", 
                                    lines=5, 
                                    max_lines=8, 
                                    interactive=False
                                )

                    recipe_load_status = gr.Textbox(label="Recipes Status", 
                                                    lines=4, 
                                                    interactive=False, 
                                                    visible=False)

                                    
                with gr.Column(scale=2): # This should be the right-side column
                    with gr.Group(elem_id="recipe_detail_panel"):
                        # Header Area
                        detail_title = gr.HTML("<h2>📜 Select a Recipe to See Details</h2>")
                        
                        with gr.Row():
                            detail_source = gr.Textbox(label="Source", interactive=False)
                            detail_score = gr.Number(label="Spoonacular Score", interactive=False)
                            detail_price = gr.Number(label="Price Per Serving ($)", interactive=False)
                        
                        detail_dishtypes = gr.Textbox(label="Dish Types", interactive=False, lines=3)
                        detail_summary = gr.HTML(label="Summary") # Summary contains HTML tags
                        
                        # Ingredients & Instructions
                        with gr.Row():
                            with gr.Column(scale=2):
                                    detail_extended_ingredients = gr.Textbox(
                                        label="Extended Ingredients", 
                                        lines=10, 
                                        max_lines=15, 
                                        interactive=False
                                    )
                                
                            # NEW: Add the image here, to the right of ingredients
                            with gr.Column(scale=2):
                                detail_recipe_image = gr.Image(
                                    label="Recipe Image", 
                                    show_label=False, 
                                    interactive=False,
                                    elem_id="recipe_main_img"
                                )

                        detail_instructions = gr.Markdown(label="Cooking Instructions")

                        # Wine & Meta
                        with gr.Row():
                            detail_wine = gr.Textbox(label="Wine Pairings", interactive=False)
                            detail_cuisines = gr.Textbox(label="Cuisines", interactive=False)
                        
                        # Taste, Diets, and Nutrition
                        with gr.Row():
                            with gr.Column(scale=2, min_width=200):
                                detail_taste = gr.Label(label="Taste Profile")
                            with gr.Column(scale=1, min_width=140):
                                detail_diets = gr.Textbox(label="Diets", interactive=False)
                            with gr.Column(scale=2, min_width=350):    
                                detail_nutrition = gr.Textbox(label="Nutrition Summary", lines=5, interactive=False)
                                


                use_ocr_scanning_cb.change(
                    fn=update_ocr_scanning_flag, 
                    inputs=[use_ocr_scanning_cb])                  

                classify_ingredients_btn.click(
                    fn=classify_ingredient,
                    inputs=[input_image],
                    outputs=[classified_ingredient_name, confidence_level]
                )

                add_classified_btn.click(
                    fn=add_classified_ingredient,
                    inputs=[classified_ingredient_name, input_image],
                    outputs=[selected_ingredients_gallery, input_image, classified_ingredient_name, confidence_level]
                )                                                              

                add_standard_ingredients_btn.click(
                    fn=add_standard_ingredients,
                    inputs=[meat_dropdown, fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown],
                    outputs=[selected_ingredients_gallery, meat_dropdown, fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown]
                )

                clear_standard_ingredients_btn.click(
                    fn=lambda: (gr.update(value=None), 
                                gr.update(value=None),
                                gr.update(value=[]),
                                gr.update(value=None),
                                gr.update(value=[])),
                    outputs=[meat_dropdown, fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown]
                )                                 
              
                add_custom_ingredients_btn.click(
                    fn=add_custom_ingredients,
                    inputs=[custom_ingredients],
                    outputs=[selected_ingredients_gallery, custom_ingredients]                    
                )

                clear_custom_ingredients_btn.click(
                    fn=lambda: (''),
                    outputs=[custom_ingredients]
                )                                        

                translate_to_spoonacular_btn.click(
                    fn=translate_ingredient_names_to_spoonacular,
                    outputs=[selected_ingredients_gallery, llm_results_dataframe]
                )

                selected_ingredients_gallery.select(fn=remove_ingredient_from_gallery, 
                                       outputs=[selected_ingredients_gallery]) 

                clear_all_ingredients_btn.click(
                    fn=clear_all_ingredients,
                    outputs=[input_image, classified_ingredient_name, confidence_level, meat_dropdown,
                             fish_dropdown, shellfish_dropdown, pasta_dropdown, spice_dropdown, 
                             custom_ingredients, selected_ingredients_gallery, llm_results_dataframe]
                )                     

            spoonacular_ingredients_to_recipe_btn.click(
                    fn=load_ingredients_into_recipe_management,
                    outputs=[selected_ingredients_gallery]
                )                                

            def toggle_selected_ingredient(data: gr.SelectData):
                global __current_ingredients, __current_ingredient_gallery_items, __ingredient_gallery_items
                
                # Get the specific item clicked from the master pantry
                clicked_item = __ingredient_gallery_items[data.index] 

                # If this is the only item in the list, remove it
                if len(__current_ingredient_gallery_items) == 1 and __current_ingredient_gallery_items[0] == __DEFAULT_LOADING_INGREDIENTS:
                    __current_ingredient_gallery_items = [] 
                    __current_ingredients = []
                
                # Toggle Logic: If it's there, remove it. If not, add it.
                ingredient_name = clicked_item[1]
                if clicked_item in __current_ingredient_gallery_items:
                    __current_ingredient_gallery_items.remove(clicked_item)
                    __current_ingredients.remove(ingredient_name)
                else:
                    __current_ingredient_gallery_items.append(clicked_item)
                    __current_ingredients.append(ingredient_name)

                # Check if list is empty; if so add place holder
                if not __current_ingredient_gallery_items:
                    __current_ingredient_gallery_items = [__DEFAULT_LOADING_INGREDIENTS]
                
                # Return the updated list to the SECOND gallery
                return __current_ingredient_gallery_items, __current_ingredient_gallery_items
            
            def toggle_selected_recipe_ingredient(data: gr.SelectData):
                global __current_ingredient_gallery_items, __ingredient_gallery_items
                
                # Get the specific item clicked from the master pantry
                clicked_item = __current_ingredient_gallery_items[data.index] 

                # If this is the only item in the list, remove it
                if len(__current_ingredient_gallery_items) == 1:
                    __current_ingredient_gallery_items = [__DEFAULT_LOADING_INGREDIENTS]
                else:
                    __current_ingredient_gallery_items.remove(clicked_item)

                # Return the updated list to the SECOND gallery
                return __current_ingredient_gallery_items, __current_ingredient_gallery_items
            
            ingredients_gallery.select(fn=toggle_selected_ingredient, 
                                       outputs=[recipe_ingredients_gallery, selected_ingredients_gallery])   
            
            recipe_ingredients_gallery.select(fn=toggle_selected_recipe_ingredient, 
                                       outputs=[recipe_ingredients_gallery, selected_ingredients_gallery])                         

            get_recipes_btn.click(
                fn=get_recipes,
                outputs=[recipe_load_status],
            )

            download_recipe_images_btn.click(
                fn=download_recipe_images,
                outputs=[recipe_load_status, recipe_gallery],
            )   

            load_recipe_button.click(
                fn=load_recipe_details,
                inputs=[selected_recipe_id, selected_recipe_name],
                outputs=[
                    detail_title, detail_source, detail_score, detail_price, 
                    detail_dishtypes, detail_summary, 
                    detail_extended_ingredients, 
                    detail_recipe_image,
                    detail_instructions, 
                    detail_wine, detail_cuisines, 
                    detail_taste, detail_diets, 
                    detail_nutrition
                ]
            )         

            reset_ingredients_btn.click(
                fn=reset_selected_ingredients,
                inputs=[],
                outputs=[recipe_ingredients_gallery, selected_ingredients_gallery]
            )

            clear_ingredients_btn.click(
                fn=clear_ingredients,
                inputs=[],
                outputs=[ingredient_load_status],
            )            

            clear_recipes_btn.click(
                fn=clear_recipes_ui,
                inputs=[],
                outputs=[
                    # Galleries & Status
                    recipe_gallery, recipe_load_status,
                    
                    # Center Summary Panel
                    recipe_title_display, recipe_likes_display, 
                    missing_count_display, missing_ingredients_list, 
                    unused_count_display, unused_ingredients_list,
                    selected_recipe_id, selected_recipe_name,
                    
                    # Right Detail Panel
                    detail_title, 
                    detail_source, detail_score, detail_price, 
                    detail_dishtypes, detail_summary, 
                    detail_extended_ingredients, detail_recipe_image, detail_instructions, 
                    detail_wine, detail_cuisines, 
                    detail_taste, detail_diets, detail_nutrition
                ]
            )

            def on_recipe_select(data: gr.SelectData):
                global __current_recipes

                # Get the selected recipe id
                recipe_id = __recipes_id_map.get(data.index)
                
                # Use data.index to find the recipe
                selected_recipe = __current_recipes[data.index]
                recipe_name = selected_recipe.get('title', "Unknown")

                # 1. Extract and format Missing Ingredients
                missing_objs = selected_recipe.get('missedIngredients', [])
                missing_names = [obj['name'] for obj in missing_objs]
                missing_count = len(missing_names)
                missing_text = "\n".join(missing_names) if missing_names else "None!"

                # 2. Extract and format Unused Ingredients
                unused_objs = selected_recipe.get('unusedIngredients', [])
                unused_names = [obj['name'] for obj in unused_objs]
                unused_count = len(unused_names)
                unused_text = "\n".join(unused_names) if unused_names else "None"

                # 3. Get title and likes
                title = selected_recipe.get('title', "Unknown")
                likes = selected_recipe.get('likes', 0)

                return title, likes, missing_count, missing_text, unused_count, unused_text, recipe_id, recipe_name
            
            recipe_gallery.select(
                fn=on_recipe_select,
                outputs=[
                    recipe_title_display, 
                    recipe_likes_display, 
                    missing_count_display, 
                    missing_ingredients_list, 
                    unused_count_display, 
                    unused_ingredients_list,
                    selected_recipe_id,
                    selected_recipe_name
                ]
            )            



if __name__ == "__main__":
    # Add your specific image directories to the allowed_paths list
    pantry_to_table.launch(
        allowed_paths=["recipes_images", "recipes_scaled_images", __INGREDIENTS_IMAGES_FOLDER, "assets"]
    )
