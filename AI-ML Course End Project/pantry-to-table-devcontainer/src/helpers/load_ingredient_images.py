"""Utility function for loading ingredient images for model building, tuning and testing.

    Open Images V7: Large collection of 9 million photos with pre-made labels for thousands of objects.

    FiftyOne: Python tool used to search, filter, and download only the specific images and labels you need.
"""

import os
import re
import time
import random
import shutil
import cv2
import yaml
import numpy as np
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

# TEST_INGREDIENTS_SET = ['Carrot', 'Tomato', 'Cabbage', 'Potato', 'Chicken', 'Pasta', 'Bell pepper', 'Zucchini', 'Lemon', 'Cheese']

TEST_INGREDIENTS_SET = ['Carrot', 'Broccoli', 'Bell pepper', 'Tomato', 'Lemon',
                        'Corn', 'Eggplant', 'Cucumber', 'Onion', 'Potato',
                        'Mushroom', 'Avocado']

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

MAX_IMAGE_SAMPLES      = 20000      # Raw download pool per ingredient
SAMPLES_PER_INGREDIENT = 500        # Final "Clean" count for training

PANTRY_INGREDIENTS_IMAGES_FOLDER = '../../downloaded_images/pantry_ingredients_images'
DATA_YAML_FILE_FOLDER            = '../../notebooks/pantrydata'
TEMPORARY_RAW_IMAGES_FOLDER      = '../../downloaded_images/temp_raw_downloads'

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
                    # Specific check: if we are purging temp_raw_downloads, 
                    # we want to ensure all sub-images are gone before recreation
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


def normalize_name(name):
    """
    Standardizes ingredient names for reliable mapping.
    """
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'[-_]', ' ', name)
    return name.strip()

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

    # 2. Setup Roboflow & Normalized Master Map
    rf = Roboflow(api_key=api_key)
    
    master_map = {normalize_name(name): i for i, name in enumerate(ingredients)}
    unresolved_classes = {}

    source_projects = [
        {"id": "vegetables/vegetables-el4g6",                                       "version": 1},
        {"id": "sqdq/fruits-and-vegetables-qlxmk",                                  "version": 1},
        {"id": "zzigmug/fruits-and-vegetables-knetf",                               "version": 1},
        {"id": "yaman-e/food-ingredients-detection-qfit7",                          "version": 48}, 
        {"id": "lhu-dqyuc/fruits-and-vegetables-mfsau",                             "version": 1},
        {"id": "college-74jj5/freshness-fruits-and-vegetables",                     "version": 1},
        {"id": "cse299/fruit-and-vegetable",                                        "version": 1},
        {"id": "project-6000-agriculture/fruits-and-vegetables-17y0t",              "version": 1},
        {"id": "michael-ringer/freiburg-groceries",                                 "version": 10},

        # Additional Sources to try to capture fruit/vegetables in their raw state
        # 24,583 images, CC BY 4.0
        # Classes: Carrot, Cabbage, Cucumber, Eggplant, Onion, Potato, Tomato
        {"id": "nutrilens-qvsz6/food-ingredients-detection-nxe34",                  "version": 3},

        # 7,952 images, CC BY 4.0
        # Classes: carrot, cucumber, eggplant, onion, potato, tomato, corn, bell pepper
        # Confirmed raw whole vegetables from image thumbnails
        {"id": "test-on9hk/vegetables-kacga",                                       "version": 5},

        # 9,780 images, license UNCONFIRMED — verify before training use
        # Classes: Carrot, Broccoli, Tomato, Potato, Onion, Cucumber, Corn, Avocado, Mushroom
        {"id": "food-recipe-ingredient-images-0gnku/food-ingredients-dataset",      "version": 4},
    ]    

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMPORARY_RAW_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(DATA_YAML_FILE_FOLDER, exist_ok=True)

    temp_raw_dir = TEMPORARY_RAW_IMAGES_FOLDER
    purge_pantry_workspace([temp_raw_dir, output_dir])

    all_data_pool = [] 

    # 3. Download and Aggregate Data
    for proj in source_projects:
        project_id = proj['id']
        print(f"--- Fetching source: {project_id} ---")
        
        try:
            project_obj = rf.project(project_id)
            version_obj = project_obj.version(proj['version'])
            
            dataset = version_obj.download(
                "yolov8", 
                location=os.path.join(temp_raw_dir, project_id.replace('/', '_'))
            )

            # --- FIXED: Build translation_map from the downloaded data.yaml ---
            with open(os.path.join(dataset.location, "data.yaml"), 'r') as yf:
                local_yaml = yaml.safe_load(yf)
                local_names = local_yaml.get('names', [])
            
            name_list = local_names if isinstance(local_names, list) else list(local_names.values())
            translation_map = {i: master_map[normalize_name(n)] 
                              for i, n in enumerate(name_list) if normalize_name(n) in master_map}

            # SCANNING LOGIC
            for root, dirs, files in os.walk(dataset.location):
                if os.path.basename(root) == "images":
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and len(all_data_pool) < pool_size:
                            img_path = os.path.join(root, f)
                            lbl_path = os.path.join(os.path.dirname(root), "labels", os.path.splitext(f)[0] + ".txt")
                            
                            if os.path.exists(lbl_path):
                                with open(lbl_path, 'r') as lf:
                                    lines = [l.strip() for l in lf.readlines() if l.strip()]
                                
                                if len(lines) == 1:
                                    try:
                                        source_id = int(lines[0].split()[0])
                                        if source_id in translation_map:
                                            master_id = translation_map[source_id]
                                            all_data_pool.append((img_path, lbl_path, master_id))
                                    except (ValueError, IndexError):
                                        continue
        
        except Exception as e:
            print(f"⚠️ Skipping {project_id} due to error: {e}")

    # 4. Validation
    if not all_data_pool:
        print("❌ CRITICAL ERROR: No image/label pairs found.")
        return

    # 5. Balancing & Augmentation
    print(f"\n⚖️ Balancing Dataset: Aiming for {final_size} images per ingredient...")
    random.shuffle(all_data_pool)
    
    class_map = {i: [] for i in range(len(ingredients))}
    for img_p, lbl_p, master_id in all_data_pool:
        if len(class_map[master_id]) < final_size:
            class_map[master_id].append((img_p, lbl_p, master_id))

    balanced_pool = []
    for class_id, pairs in class_map.items():
        current_count = len(pairs)
        balanced_pool.extend(pairs)
        
        if 0 < current_count < final_size:
            num_needed = final_size - current_count
            print(f"🔄 Augmenting {ingredients[class_id]}: {current_count} -> {final_size}")
            for i in range(num_needed):
                src_img_p, src_lbl_p, m_id = random.choice(pairs)
                ext = os.path.splitext(src_img_p)[1]
                
                # Maintain identical base naming for augmented file pairs
                aug_base = os.path.splitext(os.path.basename(src_img_p))[0] + f"_aug_{i}"
                aug_img_p = os.path.join(os.path.dirname(src_img_p), aug_base + ext)
                aug_lbl_p = os.path.join(os.path.dirname(src_lbl_p), aug_base + ".txt")

                img = cv2.imread(src_img_p)
                if img is None: continue
                if random.random() > 0.5: img = cv2.flip(img, 1) 
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
                hsv[:,:,2] *= random.uniform(0.8, 1.2) 
                hsv[:,:,2][hsv[:,:,2] > 255] = 255
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                cv2.imwrite(aug_img_p, img)
                shutil.copy(src_lbl_p, aug_lbl_p)
                balanced_pool.append((aug_img_p, aug_lbl_p, m_id))

    all_data_pool = balanced_pool
    # --- FIXED: 70/15/15 SPLIT LOGIC ---
    train_val, test = train_test_split(all_data_pool, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1764, random_state=42) # 0.15 / 0.85 approx 0.1764
    split_map = {'train': train, 'val': val, 'test': test}

   # 6. Export to Unified YOLO Structure
    print(f"Merging data into {output_dir}...")
    for split_name, pairs in split_map.items():
        img_dest = os.path.join(output_dir, split_name, "images")
        lbl_dest = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for i, (img_path, lbl_path, master_id) in enumerate(pairs):
            # --- FIXED: Simplified Naming (classname-uniqueid.ext) ---
            class_name = ingredients[master_id].lower().replace(" ", "_")
            unique_base = f"{class_name}-{i}"
            ext = os.path.splitext(img_path)[1]
            
            shutil.copy(img_path, os.path.join(img_dest, unique_base + ext))
            
            with open(os.path.join(lbl_dest, unique_base + ".txt"), 'w') as nf:
                with open(lbl_path, 'r') as of:
                    for line in of:
                        parts = line.split()
                        if not parts: continue
                        parts[0] = str(master_id)
                        nf.write(" ".join(parts) + "\n")

    # 7. Generate Master data.yaml
    yaml_path = os.path.join(DATA_YAML_FILE_FOLDER, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n\nnames:\n")
        for i, name in enumerate(ingredients):
            f.write(f"  {i}: {name}\n")

    # 8. Cleanup
    shutil.rmtree(temp_raw_dir)

    print('─────────────────────────────────────────────────────────────────────────')
    print(f"\n✅ DATASET READY: {output_dir}")
    print(f"Stats: Train({len(train)}) | Val({len(val)}) | Test({len(test)})")
    print('─────────────────────────────────────────────────────────────────────────')

    elapsed = time.time() - imageload_start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f'Total Image Load and Merge runtime: {hours}h {minutes}m {seconds}s')
    print('─────────────────────────────────────────────────────────────────────────')
    
def load_pantry_images_from_open_images():

    purge_pantry_workspace([TEMPORARY_RAW_IMAGES_FOLDER, PANTRY_INGREDIENTS_IMAGES_FOLDER])

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
