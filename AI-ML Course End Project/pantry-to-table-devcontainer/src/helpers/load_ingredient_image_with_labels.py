"""Utility function for loading ingredient images with product labels

    OpenFoodFacts - food product database with images of labeled food
    Targeting - English [US, Canada]
    URL: https://world.openfoodfacts.org

    Images can be filtered by categories.
    The desired categories are similar to the downloaded non-labeled images:
        VEGETABLES
        FRUIT
        MEAT/PROTEINS
        SEAFOOD
        DAIRY
        GRAINS/PASTAS
        CANNED GOODS
        PANTRY_INGREDIENTS

    OpenFoodFacts API expects category filters to be in lower case.
    The mapping from the PantryToTable categores to the OpenFoodFacts labels:

    VEGETABLES          =>  vegetables          --  fresh/frozen veggies.
    FRUIT               =>  fruits              --  Standard pluralized tag.
    MEAT/PROTEINS       =>  meats               --  You can also use poultry for more specific meat types.
                        =>  poultry
    SEAFOOD             =>  seafood             --  Includes fish, crustaceans, and mollusks.
    DAIRY               =>  dairies             --  milk, cheese, and yogurt.
    GRAINS/PASTAS       =>  pastas              --  pastas is very specific; cereals-and-their-products is the technical "grains" tag.
                        =>  cereals          
    CANNED GOOD         =>  Scanned-foods       --  canned beans, fruit, vegetables to soups.
    PANTRY_INGREDIENTS  =>  groceries           --  "catch-all" tag for dry goods and shelf-stable items.    
"""

import os
import requests
import time
import re
import shutil


INGREDIENT_CATEGORIES = ['vegetables', 'fruits', 'meats', 'poultry', 'seafood', 
                         'dairies', 'pastas', 'canned-foods', 'groceries']


def normalize_filename(name):
    """Standardizes names for file safety."""

    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    return name.strip('_')

def load_ingredient_image_with_labels(limit_per_cat=100, target_folder="pantry_ocr_dataset"):
    """
    Downloads package images and ground-truth text into category-specific subdirectories.
    Structure: target_folder/category/images and target_folder/category/labels
    """

    # DELETE existing root folder to ensure a clean state
    if os.path.exists(target_folder):
        print(f"🧹 Purging existing directory: {target_folder}")
        shutil.rmtree(target_folder)
    
    os.makedirs(target_folder)

    search_url = "https://world.openfoodfacts.org/cgi/search.pl"
    headers = {'User-Agent': 'PantryToTableProject - Student Research - (yourname@email.com)'}
    
    total_downloaded = 0

    # Assume INGREDIENT_CATEGORIES is defined globally or passed in
    for category in INGREDIENT_CATEGORIES:
        print(f"📂 Processing Category: {category.upper()}")
        
        # Setup Category-Specific Subdirectories
        cat_dir = os.path.join(target_folder, category)
        images_dir = os.path.join(cat_dir, "images")
        labels_dir = os.path.join(cat_dir, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for country in ["us", "ca"]:
            params = {
                "action": "process",
                "tagtype_0": "categories",
                "tag_contains_0": "contains",
                "tag_0": category,
                "cc": country,
                "lc": "en",
                "page_size": limit_per_cat,
                "json": 1,
                "fields": "product_name,image_front_url"
            }

            # --- RETRY LOGIC FOR 503 ERRORS ---
            data = {"products": []}
            for attempt in range(3):
                try:
                    response = requests.get(search_url, params=params, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        break
                    elif response.status_code == 503:
                        wait = (attempt + 1) * 5
                        print(f"⚠️ Server Busy (503). Waiting {wait}s...")
                        time.sleep(wait)
                except Exception:
                    time.sleep(2)

            # --- DOWNLOAD AND SORTING LOOP ---
            for product in data.get('products', []):
                img_url = product.get('image_front_url')
                raw_name = product.get('product_name', 'unknown_ingredient')
                
                if img_url:
                    try:
                        clean_name = normalize_filename(raw_name)
                        # Format: index_category_name
                        base_filename = f"{total_downloaded}_{category}_{clean_name[:25]}"
                        
                        img_res = requests.get(img_url, timeout=10)

                        if img_res.status_code == 200:
                            # Save to category/images
                            img_path = os.path.join(images_dir, f"{base_filename}.jpg")
                            with open(img_path, 'wb') as f:
                                f.write(img_res.content)
                            
                            # Save to category/labels
                            label_path = os.path.join(labels_dir, f"{base_filename}.txt")
                            with open(label_path, 'w', encoding='utf-8') as f:
                                f.write(raw_name)
                            
                            total_downloaded += 1
                            time.sleep(0.1) # Polite delay

                    except Exception:
                        continue

    print('─────────────────────────────────────────────────────────────────────────')
    print(f"✅ DATASET READY: {target_folder}")
    print(f"Structure: [category]/images and [category]/labels")
    print(f"Total Samples Across All Categories: {total_downloaded}")
    print('─────────────────────────────────────────────────────────────────────────')


if __name__ == "__main__":
    # Categories based on your specific requirements

    load_ingredient_image_with_labels(limit_per_cat=100, 
                                      target_folder='download_images_with_labels')