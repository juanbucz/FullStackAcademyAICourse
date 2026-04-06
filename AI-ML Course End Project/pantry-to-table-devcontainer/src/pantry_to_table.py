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
"""

import os
import sys
import logging
import time
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

from utilities.utilities import utilities as utils
from utilities.spoonacular_utilities import spoonacular_utilities as su

os.environ['TF_USE_LEGACY_KERAS'] = '1'    # must be first

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

__RECIPES_SCALED_IMAGE_DIR = 'recipes_scaled_images'

# ---------------------------------------------------------------------------
# Create Default Gallery Image once instead of every time from URL
# ---------------------------------------------------------------------------
def create_placeholder():

    # Create a 150x150 gray square
    img = Image.new('RGB', (150, 150), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,70), "Loading...", fill=(255,255,0))
    
    # Save to assets folder
    os.makedirs("assets", exist_ok=True)
    img.save('assets/loading.png')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
__loaded_ingredients = []
__current_ingredients = []
__current_recipes = []

create_placeholder()
__ingredient_gallery_items =[__DEFAULT_LOADING_INGREDIENTS]
__current_ingredient_gallery_items =[__DEFAULT_LOADING_INGREDIENTS]

__recipe_gallery_items =[__DEFAULT_LOADING_RECIPES]

__ingredients_id_map = {}
__recipes_id_map = {}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_ingredients() -> str:
    """Call the Spoonacular API to load the ingredients."""
    global __loaded_ingredients, __ingredient_gallery_items, __ingredients_id_map

    results = su.load_Spoonacular_ingredients()

    # Load ingredients for display/selection
    __loaded_ingredients = su.all_ingredients

    # For 'List Box'
    __ingredient_gallery_items = [(f"ingredients_images/{item['image']}", item['name']) for item in __loaded_ingredients]

    # For retrieving ingredient ID
    __ingredients_id_map = {index: item['id'] for index, item in enumerate(__loaded_ingredients)}

    return results

def download_ingredient_images() -> str:        
    """Download the ingredient images from Spoonacular"""
    global __ingredient_gallery_items

    results = su.download_ingredient_images()
    return results, __ingredient_gallery_items, __ingredient_gallery_items

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

def load_recipe() -> tuple[str, any]:
    """Load recipe details for the selected Spoonacular recipe."""
    print()

def clear_recipes() -> tuple[str, any]:
    """Delete all the loaded Spoonacular recipes."""
    print()
    
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

    gr.Markdown("""
    # This system use Spoonacular to retrieve ingredients, recipes and match ingredients with recipes
    
    **How it works:**
    1. **Manage Ingredients** - calls out to Spoonacular API to retrieve requested number ingredients and their matching images
    2. **Manage Recipes** - Based on selected ingredients retrieve a list of recipes
    """)

    with gr.Tabs():

        # ------------------------------------------------------------------
        # Tab 1: Ingredient Management
        # ------------------------------------------------------------------
        with gr.Tab("1. Manage Ingredients"):
            ingest_instructions = gr.Markdown("""
            Loads requested number of ingredients, and their related images via Spoonacular API
            """)

            # State to hold multiple selected IDs
            selected_ingredients_state = gr.State(value=[])

            with gr.Row():
                with gr.Column():

                    with gr.Row():
                        load_ingredients_btn = gr.Button("Load Spoonacular Ingredients", variant="primary")
                        download_ingredient_images_btn = gr.Button("Download Spoonacular Ingredient Images", variant="primary")
                        with gr.Column(min_width=150):
                            reset_ingredients_btn = gr.Button("Reset Recipe Ingredients", variant="stop")
                            clear_ingredients_btn = gr.Button("Clear All Ingredients", variant="stop")

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
                outputs=[ingredient_load_status],
            )

            download_ingredient_images_btn.click(
                fn=download_ingredient_images,
                outputs=[ingredient_load_status, ingredients_gallery],
            )            

            clear_ingredients_btn.click(
                fn=clear_ingredients,
                inputs=[],
                outputs=[ingredient_load_status],
            )

        # ------------------------------------------------------------------
        # Tab 2: Query
        # ------------------------------------------------------------------
        with gr.Tab("2. Manage Recipes"):
            gr.Markdown("""
            Loads top ranked recipes and their images based on selected ingredients via Spoonacular API
            """)

            with gr.Row():
                with gr.Column():

                    with gr.Row():
                        get_recipes_btn = gr.Button("Load Spoonacular Recipes", variant="primary")
                        download_recipe_images_btn = gr.Button("Download Spoonacular Recipe Images", variant="primary")
                        with gr.Column(min_width=150):
                            load_recipe_button = gr.Button("Load Selected Recipe", variant="primary")
                            clear_recipes_btn = gr.Button("Clear All Recipes", variant="stop")

                    gr.Markdown("### 📝 Available Recipe Selection")

                    selected_indices = gr.State([])

                    with gr.Row():
                            
                            with gr.Column(scale=2, min_width=300):
                                # The Selected Ingredients Gallery (Items ready for the recipe)
                                gr.Markdown("### 🛒 Selected Ingredients")
                                selected_ingredients_gallery = gr.Gallery(
                                    value=__current_ingredient_gallery_items, 
                                    elem_id="pantry_list", 
                                    columns=1, 
                                    height=600, 
                                    interactive=True
                                )

                            # The Available Recipes Gallery 
                            with gr.Column(scale=3, min_width=400):
                                gr.Markdown("### 📝 Available Recipes")
                                recipe_gallery = gr.Gallery(
                                    value=__recipe_gallery_items, 
                                    elem_id="pantry_list", 
                                    columns=1, 
                                    height=600, 
                                    interactive=True
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

                    recipe_load_status = gr.Textbox(label="Recipes Status", lines=4, interactive=False)

                                    
                with gr.Column():
                    recipe_load_status_holder = gr.Textbox(label="Recipes Status", lines=4, interactive=False)

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
                fn=load_recipe,
                outputs=[recipe_load_status],
            )         

            clear_recipes_btn.click(
                fn=clear_recipes,
                inputs=[],
                outputs=[ingredient_load_status],
            )

            def on_recipe_select(data: gr.SelectData):
                global __current_recipes
                
                # Use data.index to find the recipe
                selected_recipe = __current_recipes[data.index]
                
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

                return title, likes, missing_count, missing_text, unused_count, unused_text
            
            recipe_gallery.select(
                fn=on_recipe_select,
                outputs=[
                    recipe_title_display, 
                    recipe_likes_display, 
                    missing_count_display, 
                    missing_ingredients_list, 
                    unused_count_display, 
                    unused_ingredients_list
                ]
            )            


if __name__ == "__main__":
    pantry_to_table.launch()
