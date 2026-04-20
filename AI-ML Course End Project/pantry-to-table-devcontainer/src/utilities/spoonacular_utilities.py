# spoonacular_utilities.py  
# ─────────────────────────────────────────────
# Utility functions for working with Spoonacular
# ─────────────────────────────────────────────

import os
import shutil
import logging
from datetime import datetime
from typing import Tuple, Optional

import requests
import json
import re


from .utilities import utilities as utils


class SpoonacularUtilities:
    """Centralized utility class for the Pantry App."""

    # ─────────────────────────────────────────
    # CONSTANTS
    # ─────────────────────────────────────────

    __SPOONACULAR_INGREDIENTS_API_URL = 'https://api.spoonacular.com/food/ingredients/search'
    __SPOONACULAR_RECIPE_API_URL = 'https://api.spoonacular.com/recipes/findByIngredients'
    __SPOONACULAR_RECIPE_DETAILS_API_URL = 'https://api.spoonacular.com/recipes/{recipe_id}/information'
    __API_KEY = 'd6f1489c0e044b0a95ea2e04eb8ff3a6'
    __TOTAL_INGREDIENTS = 20
    __TOTAL_RECIPES     = 30
    __RESULTS_PER_CALL  = 100

    __INGREDIENTS_SEARCH_URL = 'https://api.spoonacular.com/food/ingredients/search'
    __INGREDIENTS_IMAGE_URL  = 'https://img.spoonacular.com/ingredients_'
    __INGREDIENTS_IMAGE_DIR  = 'ingredients_images'
    __INGREDIENTS_IMAGE_SIZE = '100x100'



    __RECIPES_IMAGE_URL         = 'https://img.spoonacular.com/ingredients_'
    __RECIPES_DIR               = 'recipes'
    __RECIPES_IMAGE_DIR         = 'recipes_images'
    __RECIPES_SCALED_IMAGE_DIR  = 'recipes_scaled_images'
    __RECIPES_SCALED_IMAGE_SIZE = (100,100)
    __RECIPE_DEFAULT_IMAGE      = 'default_images/default_recipe.jpg'
    

    # ─────────────────────────────────────────
    # Member variables
    # ─────────────────────────────────────────
    __total_ingredients = __TOTAL_INGREDIENTS
    all_ingredients        = []
    current_recipes        = []
    current_recipe_details = None

    # ─────────────────────────────────────────
    # Spoonacular API HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def verify_ingredient(ingredient_name) -> str:
        """
            Checks if an ingredient exists in the Spoonacular database.
            Returns simple True or False
        """

        su = SpoonacularUtilities  # alias

        url = utils.build_url(
                        su.__INGREDIENTS_SEARCH_URL,
                        query  = ingredient_name,
                        number = 5,
                        apiKey = su.__API_KEY
                    )
        
        try:
            results = requests.get(url)

            # Check if any results were returned
            if not results: 
                return None
            
            data = results.json()
            ingredient_list = data.get('results', [])
            names = [item['name'] for item in ingredient_list]
            
            # Return the "official" name from their DB (e.g., "Chicken Breasts")
            # Have to account for Spoonacular being TOO SMART. Just want closest match.
            # Logic: Look for an exact match first
            for name in names:
                if name.lower() == ingredient_name.lower():
                    return name

            # Fallback: Return the shortest name (usually the most "basic" form)
            # This prevents getting "sliced chicken breast" if "chicken breast" is an option
            return min(names, key=len)
            
        except Exception as e:
            print(f"Error verifying ingredient: {e}")
            return None        


    @staticmethod
    def load_Spoonacular_ingredients() -> str:
        """Call Spoonacular Ingredients API using utility helper methods and return consolidated status."""

        su = SpoonacularUtilities  # alias
        
        # Track the final status message
        api_status_msg = ''
        accumulated_ingredients = []

        for offset in range(0, su.__TOTAL_INGREDIENTS, su.__RESULTS_PER_CALL):
            url = utils.build_url(
                                    su.__SPOONACULAR_INGREDIENTS_API_URL,
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
    def download_ingredient_images(ingredients_list=all_ingredients, path=__INGREDIENTS_IMAGE_DIR) -> str:
        """
        Download and save images associated with loaded ingredients
        This expects a list of tupled ingrediengs (item_image, item_name)
        """

        su = SpoonacularUtilities          # alias

        status = []

        os.makedirs(path, exist_ok=True)
        results = utils.clear_directory(path)
        utils.logger.info(results)
        status.append(results)

        for item in ingredients_list:
            # 1. Construct the URL

            # [item_image, item_name] e.g., 'apple.jpg'
            # image_name => image_path/image_name
            # Split to get image name
            img_name = None
            if '/' in item[0]:
                img_name = item[0].split('/')[1]
            else:
                img_name = item[0]

            img_url = f'{su.__INGREDIENTS_IMAGE_URL}{su.__INGREDIENTS_IMAGE_SIZE}/{img_name}'
            
            # 2. Define local save path
            save_path = os.path.join(path, img_name)
            
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
    def download_ingredient_images(ingredients_to_download: list, path=__INGREDIENTS_IMAGE_DIR) -> set:
        """
        Returns a SET of standardized names that are successfully available on disk.
        """

        su = SpoonacularUtilities
        successful_downloads = set()
        os.makedirs(path, exist_ok=True)
        
        for ingredient in ingredients_to_download:
            img_name = f"{ingredient}.jpg"
            save_path = os.path.join(path, img_name)
            
            # If it already exists, consider it a success
            if os.path.exists(save_path):
                successful_downloads.add(ingredient)
                continue

            try:
                img_url = f'{su.__INGREDIENTS_IMAGE_URL}{su.__INGREDIENTS_IMAGE_SIZE}/{img_name}'
                response = requests.get(img_url, stream=True, timeout=5)

                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    successful_downloads.add(ingredient)
                    
            except Exception:
                pass # Error handling handled by the set check in caller

        return successful_downloads
        
    
    @staticmethod
    def load_Spoonacular_recipes(ingredients_list) -> str:
        """Call Spoonacular Recipes API using utility helper methods and return consolidated status."""
        
        su = SpoonacularUtilities  # alias

        os.makedirs(su.__RECIPES_DIR, exist_ok=True)
        
        # Track the final status message
        api_status_msg = ''
        
        # Join into a comma-separated string; coming in as list of names
        ingredients_str = ",".join(ingredients_list)

        accumulated_recipes = [] 

        # 1 = maximize used ingredients; 2 = minimize missing
        # Ignores common items like water, salt, oil
        url = utils.build_url(
                                su.__SPOONACULAR_RECIPE_API_URL,
                                ingredients  = ingredients_str,
                                number       = su.__TOTAL_RECIPES,
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
        save_path = os.path.join(su.__RECIPES_DIR, f'current_recipes_{su.__TOTAL_RECIPES}.json')

        # Save to disk
        with open(save_path, 'w') as f:
            json.dump(su.current_recipes, f)

        # Return the three required strings joined by newlines or spaces
        return (
            f'{api_status_msg}\n'
            f'Downloaded {count} recipes\n'
            f'Process complete. Saved to {save_path}'
        )
    
    @staticmethod
    def download_recipe_images() -> str:
        """Download and save images associated with loaded ingredients"""
        
        su = SpoonacularUtilities          # alias

        status = []

        os.makedirs(su.__RECIPES_IMAGE_DIR, exist_ok=True)
        results = utils.clear_directory(su.__RECIPES_IMAGE_DIR)
        os.makedirs(su.__RECIPES_SCALED_IMAGE_DIR, exist_ok=True)
        results = utils.clear_directory(su.__RECIPES_SCALED_IMAGE_DIR)        

        utils.logger.info(results)
        status.append(results)

        for item in su.current_recipes:
            # 1. Construct the URL
            img_name = item['title'].replace(" ", "").replace("&", "")

            # Use regex to remove ANY non-alphanumeric characters except dots or dashes
            # This kills slashes (/), colons (:), and backslashes (\)
            img_name = re.sub(r'[^a-zA-Z0-9\-_]', '', img_name)          

            img_url = item['image']  # e.g., 'apple.jpg'
            #img_url = f'{su.__RECIPES_IMAGE_URL}{su.__RECIPES_SCALED_IMAGE_SIZE}/{img_name}'
            
            # 2. Define local save path
            save_path = os.path.join(su.__RECIPES_IMAGE_DIR, f'{img_name}.{item['imageType']}')
            
            # 3. Download and Save
            try:
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    status.append(f'Downloaded: {img_name}')
                else:
                    status.append(f'Failed: {img_name} (Status: {response.status_code})')
                    status.append(f'Created Default image: {img_name})')
                    shutil.copy2(su.__RECIPE_DEFAULT_IMAGE, save_path)
                                                      
            except Exception as e:
                status.append(f'Error downloading {img_name}: {e}')
                status.append(f'Created Default image: {img_name})')
                shutil.copy2(su.__RECIPE_DEFAULT_IMAGE, save_path)

            # 4. Scale Recipe Image to smaller size
            try:
                # Make sure the filename includes .jpg
                scaled_image_save_path = os.path.join(su.__RECIPES_SCALED_IMAGE_DIR, f"{img_name}.jpg")

                # Now call the scale utility
                utils.scale_image(save_path, scaled_image_save_path, su.__RECIPES_SCALED_IMAGE_SIZE)

            except Exception as e:
                status.append(f'Error scaling {img_name}: {e}')                


        return "\n".join(status)
    
    @staticmethod
    def load_recipe_details(recipe_id, recipe_name) -> str:
        """Call Spoonacular Recipes API using utility helper methods and return consolidated status."""
        su = SpoonacularUtilities  # alias
        
        os.makedirs(su.__RECIPES_DIR, exist_ok=True)

        # Track the final status message
        api_status_msg = ''
        
        url = utils.build_url(
                                su.__SPOONACULAR_RECIPE_DETAILS_API_URL.format(recipe_id=recipe_id),
                                includeNutrition = True,    # Get the macros
                                addWinePairing   = True,    # Get sommelier suggestions
                                addTasteData     = True,
                                apiKey           = su.__API_KEY
                            )
            
        response = requests.get(url)
        data, message = utils.handle_response(response)
            
        if data:
            # Capture the success message from the first successful call
            if not api_status_msg:
                api_status_msg = message

                # Store recipe details (json); (Client) UI Layer will parse/format for display
                su.current_recipe_details = data
        else:
            # If a call fails, return the error message immediately
            return message

        # Save to disk
        # filename = f'current_recipes_{su.__TOTAL_RECIPES}.json'

        clean_recipe_name = recipe_name.replace(" ", "").replace("&", "")
            
        # Use regex to remove ANY non-alphanumeric characters except dots or dashes
        # This kills slashes (/), colons (:), and backslashes (\)
        clean_recipe_name = re.sub(r'[^a-zA-Z0-9\-_]', '', clean_recipe_name) 

        save_path = os.path.join(su.__RECIPES_DIR, f'{clean_recipe_name}_details.json')
        with open(save_path, 'w') as f:
            json.dump(data, f)

        # Return the three required strings joined by newlines or spaces
        return (
            f'{api_status_msg}\n'
            #f'Process complete. Saved to {filename}'
        )
    
