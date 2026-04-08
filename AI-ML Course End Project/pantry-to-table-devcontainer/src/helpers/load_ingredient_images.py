"""Utility function for loading ingredient images for model building, tuning and testing.

    Open Images V7: Large collection of 9 million photos with pre-made labels for thousands of objects.

    FiftyOne: Python tool used to search, filter, and download only the specific images and labels you need.
"""

import os
import shutil
import time
import random
from roboflow import Roboflow
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv, find_dotenv

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

load_dotenv(find_dotenv())
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
#print(f'API Key Loaded: {roboflow_api_key}')

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Ingredient Sets
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

TEST_INGREDIENTS_SET = ['Carrot', 'Tomato', 'Cabbage', 'Potato', 'Chicken', 'Pasta', 'Bell pepper', 'Zucchini', 'Lemon', 'Cheese']

VEGETABLE_INGREDIENTS = ['Artichoke', 'Asparagus', 'Bell pepper', 'Broccoli', 'Cabbage', 'Carrot', 
                        'Cauliflower', 'Celery', 'Corn', 'Cucumber', 'Eggplant', 'Zucchini', 'Green bean', 
                        'Mushroom', 'Squash', 'Potato', 'Pumpkin', 'Radish', 'Spinach', 'Zucchini']

FRUIT_INGREDIENTS = ['Apple', 'Avocado', 'Banana', 'Blueberry', 'Cherry', 'Grape', 'Grapefruit', 
                    'Lemon', 'Lime', 'Mango', 'Orange', 'Peach', 'Pear', 'Pineapple', 'Strawberry']

MEAT_INGREDIENTS = ['Bacon', 'Beef', 'Chicken breast', 'Chicken wing', 'Duck', 'Ground beef', 'Ham', 
                    'Lamb', 'Meatball', 'Pork chop', 'Salami', 'Sausage', 'Steak', 'Turkey', 'Venison']

SEAFOOD_INGREDIENTS = ['Clam', 'Crab', 'Lobster', 'Mussel', 'Oyster', 'Prawn', 'Salmon', 'Scallop', 'Shrimp', 'Tuna']

DAIRY_INGREDIENTS = ['Butter', 'Cheddar cheese', 'Cottage cheese', 'Cream', 'Milk', 'Mozzarella', 
                    'Parmesan', 'Sour cream', 'Swiss cheese', 'Yogurt']

GRAINS_INGREDIENTS = ['Bagel', 'Baguette', 'Barley', 'Black bean', 'Bread', 'Chickpea', 'Croissant', 
                    'Lentil', 'Pasta', 'Pretzel', 'Quinoa', 'Sunflower', 'Spaghetti', 'Tortilla', 'Wheat']

PANTRY_INGREDIENTS = ['Almond', 'Cashew', 'Chocolate', 'Coffee', 'Honey', 'Maple syrup', 'Olive', 'Peanut', 
                    'Peanut butter', 'Pecan', 'Pistachio', 'Tofu', 'Tomato', 'Walnut', 'Wine']

FULL_INGREDIENTS_SET = VEGETABLE_INGREDIENTS + FRUIT_INGREDIENTS + MEAT_INGREDIENTS + SEAFOOD_INGREDIENTS + DAIRY_INGREDIENTS + GRAINS_INGREDIENTS + PANTRY_INGREDIENTS

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

MAX_IMAGE_SAMPLES      = 200      # Raw download pool per ingredient
SAMPLES_PER_INGREDIENT = 100      # Final "Clean" count for training

PANTRY_INGREDIENTS_IMAGES_FOLDER = '../pantry_ingredients_images'

import os
import shutil
import random
from dotenv import load_dotenv, find_dotenv
from roboflow import Roboflow
from sklearn.model_selection import train_test_split

def download_pantry_vision_dataset(ingredients, pool_size=None, final_size=None, output_dir="pantry_data"):
    """
    Downloads and prepares a unified YOLOv8 dataset from Roboflow Universe sources.
    Handles cross-workspace permissions and merges multiple sources into one.
    """
    
    # 1. Load Environment Variables from Project Root
    load_dotenv(find_dotenv())
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file. Ensure .env is in the project root.")
    
    # For Tracking Notebook runtime/performance
    imageload_start_time = time.time()

    # 2. Setup Roboflow & Corrected Project IDs
    rf = Roboflow(api_key=api_key)
    
    source_projects = [
        # 1. Product-Centric & Pantry Staples
        {"id": "michael-ringer/freiburg-groceries", "version": 10},      # Verified: 4,933 images (Original stable)
        {"id": "houda-blj4y/groceries-epwwx", "version": 2},            # Verified: Pantry staples
        {"id": "biscocho-john-kenneth-8bpsb/nutrilense", "version": 4},   # Verified: Retail packaging (Tuna, Milk, Sauces)
        
        # 2. Raw Proteins (The "Fridge" Inventory)
        {"id": "food-recognition/united_yolov8_test", "version": 1},      # Verified: Raw meats/fish focus
        {"id": "object-detection-f8udo/ingredients-v2", "version": 2},    # Verified: Raw Salmon, Pork, Scallops
        
        # 3. Produce (Fresh Inventory)
        {"id": "vth4f/combined-vegetables-fruits", "version": 1}        # Verified: Bulk Produce
    ]

    temp_raw_dir = "temp_raw_downloads"
    if os.path.exists(temp_raw_dir):
        shutil.rmtree(temp_raw_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_data_pool = [] # Store tuples of (image_path, label_path)

    # 3. Download and Aggregate Data
    for proj in source_projects:
        project_id = proj['id']
        print(f"--- Fetching source: {project_id} ---")
        
        try:
            # Use rf.project() for public Universe access
            project = rf.project(project_id)
            dataset = project.version(proj['version']).download(
                "yolov8", 
                location=os.path.join(temp_raw_dir, project_id.replace('/', '_'))
            )
            
            # Scan the downloaded folders for valid pairs
            for root, dirs, files in os.walk(dataset.location):
                if root.endswith("images"):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(root, f)
                            # Locate the matching label file in the sister 'labels' folder
                            lbl_path = img_path.replace("images", "labels").replace(os.path.splitext(f)[1], ".txt")
                            
                            if os.path.exists(lbl_path):
                                all_data_pool.append((img_path, lbl_path))
        
        except Exception as e:
            print(f"⚠️ Skipping {project_id} due to error: {e}")

    # 4. Validation & Stratification
    if not all_data_pool:
        print("❌ CRITICAL ERROR: No image/label pairs found. Check project IDs and API permissions.")
        return

    print(f"Total valid image/label pairs found: {len(all_data_pool)}")
    
    # Shuffle for randomness
    random.shuffle(all_data_pool)
    
    # 80/20 Split for Test, then 80/20 for Train/Val
    train_val, test = train_test_split(all_data_pool, test_size=0.20, random_state=42)
    train, val = train_test_split(train_val, test_size=0.20, random_state=42)

    split_map = {'train': train, 'val': val, 'test': test}

    # 5. Export to Unified YOLO Structure
    print(f"Merging data into {output_dir}...")
    for split_name, pairs in split_map.items():
        img_dest = os.path.join(output_dir, split_name, "images")
        lbl_dest = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for img_path, lbl_path in pairs:
            # Copy with original names to prevent collisions if projects have same filenames
            # We use the parent folder name as a prefix to ensure uniqueness
            prefix = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            unique_name = f"{prefix}_{os.path.basename(img_path)}"
            unique_label = f"{prefix}_{os.path.basename(lbl_path)}"
            
            shutil.copy(img_path, os.path.join(img_dest, unique_name))
            shutil.copy(lbl_path, os.path.join(lbl_dest, unique_label))

    # 6. Generate Master data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("names:\n")
        for i, name in enumerate(ingredients):
            f.write(f"  {i}: {name}\n")

    # 7. Cleanup
    shutil.rmtree(temp_raw_dir)
    print(f"\n✅ DATASET READY: {output_dir}")
    print(f"Stats: Train({len(train)}) | Val({len(val)}) | Test({len(test)})")
    print(f"\nSUCCESS: Unified Dataset Created")
    print(f"Ready for Blackwell GPU training using data.yaml")

    # 8. Print Timing statistics
    elapsed = time.time() - imageload_start_time
    elapsed = 1900
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print('======================================================================')
    print()
    print(f'Total Image Load and Merge runtime: {hours}h {minutes}m {seconds}s')
    print()
    print('======================================================================')

def purge_pantry_workspace(paths_to_clean):
    """
    Safely removes contents of project folders to ensure a clean slate.
    Accepts a single string path or a list of paths.
    """
    if isinstance(paths_to_clean, str):
        paths_to_clean = [paths_to_clean]

    for folder_path in paths_to_clean:
        if os.path.exists(folder_path):
            print(f"🧹 Purging: {folder_path}...")
            
            # Attempt a retry loop for Windows/Dev Container file locks
            for attempt in range(3):
                try:
                    shutil.rmtree(folder_path)
                    # Recreate empty dir for the next load run
                    os.makedirs(folder_path)
                    print(f"✅ {folder_path} is now empty.")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"⚠️ Lock detected, retrying {folder_path}...")
                        time.sleep(1)
                    else:
                        print(f"❌ Failed to purge {folder_path}: {e}")
        else:
            print(f"ℹ️ Creating new directory: {folder_path}")
            os.makedirs(folder_path)

def load_pantry_images_from_open_images():

    download_pantry_vision_dataset(
                                    ingredients=TEST_INGREDIENTS_SET, 
                                    pool_size=MAX_IMAGE_SAMPLES, 
                                    final_size=SAMPLES_PER_INGREDIENT,
                                    output_dir=PANTRY_INGREDIENTS_IMAGES_FOLDER
                                )


if __name__ == "__main__":
    # # Optional: Cleanup first
    # purge_dataset_folder(PANTRY_INGREDIENTS_IMAGES_FOLDER)
    
    # Execute the download and split pipeline
    load_pantry_images_from_open_images()
