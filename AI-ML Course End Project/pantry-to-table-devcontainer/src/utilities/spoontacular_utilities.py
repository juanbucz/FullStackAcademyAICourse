# spoontacular_utilities.py  
# ─────────────────────────────────────────────
# Utility functions for working with Spoontacular
# ─────────────────────────────────────────────

import os
import logging
from datetime import datetime

import requests
import json


from .utilities import utilities as utils


class spoontacular_utilities:
    """Centralized utility class for the Pantry App."""

    # ─────────────────────────────────────────
    # CONSTANTS
    # ─────────────────────────────────────────

    __SPOONTACULAR_INGREDIENTS_API_URL = 'https://api.spoonacular.com/food/ingredients/search'
    __SPOONTACULAR_RECIPE_API_URL = 'https://api.spoonacular.com/recipes/findByIngredients'
    __API_KEY = 'd6f1489c0e044b0a95ea2e04eb8ff3a6'
    __TOTAL_INGREDIENTS = 20
    __TOTAL_RECIPES     = 30
    __RESULTS_PER_CALL  = 100

    __INGREDIENTS_IMAGE_URL = 'https://img.spoonacular.com/ingredients_'
    __INGREDIENTS_IMAGE_DIR = 'ingredients_images'
    __INGREDIENTS_IMAGE_SIZE = '100x100'

    __RECIPES_IMAGE_URL = 'https://img.spoonacular.com/ingredients_'
    __RECIPES_IMAGE_DIR = 'recipes_images'
    __RECIPES_SCALED_IMAGE_DIR = 'recipes_scaled_images'
    __RECIPES_SCALED_IMAGE_SIZE = (100,100)

    # ─────────────────────────────────────────
    # Member variables
    # ─────────────────────────────────────────
    __total_ingredients = __TOTAL_INGREDIENTS
    all_ingredients = []
    current_recipes = []
    

    # ─────────────────────────────────────────
    # SPOONTACULAR API HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def load_spoontacular_ingredients() -> str:
        """Call Spoontacular Ingredients API using utility helper methods and return consolidated status."""
        su = spoontacular_utilities  # ← alias
        
        # Track the final status message
        api_status_msg = ''
        accumulated_ingredients = []

        for offset in range(0, su.__TOTAL_INGREDIENTS, su.__RESULTS_PER_CALL):
            url = utils.build_url(
                                    su.__SPOONTACULAR_INGREDIENTS_API_URL,
                                    query  = 'a',
                                    number = su.__RESULTS_PER_CALL,
                                    offset = offset,
                                    apiKey = su.__API_KEY
                                )
            
            response = requests.get(url)
            data, message = utils.handle_response(response)
            
            if data:
                # Capture the success message from the first successful call
                api_status_msg = message
               
                accumulated_ingredients.extend(data.get('results', []))
                
                formatted_lines = [
                                f"Loaded: Name:{item['name']} id:{item['id']} image:{item['image']}" 
                                    for item in accumulated_ingredients
                                ]
                api_status_msg = api_status_msg + "\n".join(formatted_lines)

            else:
                # If a call fails, return the error message immediately
                return message

        # Update the utility class state
        su.all_ingredients = accumulated_ingredients
        count = len(su.all_ingredients)
        filename = f'pantry_ingredients_{su.__TOTAL_INGREDIENTS}.json'

        # Save to disk
        with open(filename, 'w') as f:
            json.dump(su.all_ingredients, f)

        # Return the three required strings joined by newlines or spaces
        return (
            f'{api_status_msg}\n'
            f'Downloaded {count} ingredients\n'
            f'Process complete. Saved to {filename}'
        )
    
    @staticmethod
    def download_ingredient_images() -> str:
        """Download and save images associated with loaded ingredients"""
        su = spoontacular_utilities          # ← alias

        status = []

        os.makedirs(su.__INGREDIENTS_IMAGE_DIR, exist_ok=True)
        results = utils.clear_directory(su.__INGREDIENTS_IMAGE_DIR)
        utils.logger.info(results)
        status.append(results)

        for item in su.all_ingredients:
            # 1. Construct the URL
            img_name = item['image']  # e.g., 'apple.jpg'
            img_url = f'{su.__INGREDIENTS_IMAGE_URL}{su.__INGREDIENTS_IMAGE_SIZE}/{img_name}'
            
            # 2. Define local save path
            save_path = os.path.join(su.__INGREDIENTS_IMAGE_DIR, img_name)
            
            # 3. Download and Save
            try:
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    status.append(f'Downloaded: {img_name}')
                else:
                    status.append(f'Failed: {img_name} (Status: {response.status_code})')
                                  
            except Exception as e:
                status.append(f'Error downloading {img_name}: {e}')

        return "\n".join(status)
    
    @staticmethod
    def load_spoontacular_recipes(ingredients_list) -> str:
        """Call Spoontacular Recipes API using utility helper methods and return consolidated status."""
        su = spoontacular_utilities  # ← alias
        
        # Track the final status message
        api_status_msg = ''
        
        # 1. Clean the list: Extract just the names if you're passing (path, name) tuples
        # e.g., [('assets/avocado.png', 'avocado'), ...] -> ['avocado']
        ingredient_names = [item['name'] for item in ingredients_list]
        
        # 2. Join into a comma-separated string
        ingredients_str = ",".join(ingredient_names)

        accumulated_recipes = [] 

        # 1 = maximize used ingredients; 2 = minimize missing
        # Ignores common items like water, salt, oil
        url = utils.build_url(
                                su.__SPOONTACULAR_RECIPE_API_URL,
                                ingredients  = ingredients_str,
                                number       = su.__TOTAL_INGREDIENTS,
                                ranking      = 1,
                                ignorePantry = True,
                                apiKey       = su.__API_KEY
                            )
            
        response = requests.get(url)
        data, message = utils.handle_response(response)
            
        if data:
            # Capture the success message from the first successful call
            if not api_status_msg:
                api_status_msg = message
                accumulated_recipes = data

                formatted_lines = [
                    f"Loaded: Name:{item['title']} Id:{item['id']} Image:{item['image']} Used:{item['usedIngredientCount']} Missing:{item['missedIngredientCount']}" 
                        for item in accumulated_recipes
                    ]

                api_status_msg = api_status_msg + "\n".join(formatted_lines)
        else:
            # If a call fails, return the error message immediately
            return message

        # Update the utility class state
        su.current_recipes = accumulated_recipes
        count = len(su.current_recipes)
        filename = f'current_recipes_{su.__TOTAL_RECIPES}.json'

        # Save to disk
        with open(filename, 'w') as f:
            json.dump(su.current_recipes, f)

        # Return the three required strings joined by newlines or spaces
        return (
            f'{api_status_msg}\n'
            f'Downloaded {count} recipes\n'
            f'Process complete. Saved to {filename}'
        )
    
    @staticmethod
    def download_recipe_images() -> str:
        """Download and save images associated with loaded ingredients"""
        su = spoontacular_utilities          # ← alias

        status = []

        os.makedirs(su.__RECIPES_IMAGE_DIR, exist_ok=True)
        results = utils.clear_directory(su.__RECIPES_IMAGE_DIR)
        os.makedirs(su.__RECIPES_SCALED_IMAGE_DIR, exist_ok=True)
        results = utils.clear_directory(su.__RECIPES_SCALED_IMAGE_DIR)        

        utils.logger.info(results)
        status.append(results)

        for item in su.current_recipes:
            # 1. Construct the URL
            img_name = item['title']
            img_url = item['image']  # e.g., 'apple.jpg'
            #img_url = f'{su.__RECIPES_IMAGE_URL}{su.__RECIPES_SCALED_IMAGE_SIZE}/{img_name}'
            
            # 2. Define local save path
            save_path = os.path.join(su.__RECIPES_IMAGE_DIR, img_name)
            scaled_image_save_path = os.path.join(su.__RECIPES_SCALED_IMAGE_DIR, img_name)
            
            # 3. Download and Save
            try:
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    status.append(f'Downloaded: {img_name}')
                else:
                    status.append(f'Failed: {img_name} (Status: {response.status_code})')
                                  
            except Exception as e:
                status.append(f'Error downloading {img_name}: {e}')

            # 4. Scale Recipe Image to smaller size
            try:
                # Make sure the filename includes .jpg
                recipe_name = item['title'].replace("/", "-") # Clean name to avoid path errors
                scaled_image_save_path = os.path.join("recipes_scaled_images", f"{recipe_name}.jpg")

                # Now call the scale utility
                utils.scale_image(save_path, scaled_image_save_path, su.__RECIPES_SCALED_IMAGE_SIZE)

            except Exception as e:
                status.append(f'Error scaling {img_name}: {e}')                


        return "\n".join(status)
