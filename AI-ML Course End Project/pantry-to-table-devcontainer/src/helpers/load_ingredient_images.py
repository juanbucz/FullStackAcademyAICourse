"""Utility function for loading ingredient images for model building, tuning and testing.

    Open Images V7: Large collection of 9 million photos with pre-made labels for thousands of objects.

    FiftyOne: Python tool used to search, filter, and download only the specific images and labels you need.
"""

import os
import shutil
import time
import random
import cv2
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

PANTRY_INGREDIENTS_IMAGES_FOLDER = '../../downloaded_images/pantry_ingredients_images'
TEMPORARY_RAW_IMAGES_FOLDER      = '../../downloaded_images/temp_raw_downloads'

import os
import shutil
import random
from dotenv import load_dotenv, find_dotenv
from roboflow import Roboflow
from sklearn.model_selection import train_test_split

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
        {"id": "michael-ringer/freiburg-groceries", "version": 10},     # Verified: 4,933 images (Original stable)
        {"id": "houda-blj4y/groceries-epwwx", "version": 2},            # Verified: Pantry staples
        {"id": "biscocho-john-kenneth-8bpsb/nutrilense", "version": 1}, # Verified: Retail packaging (Tuna, Milk, Sauces)
        
        # 2. Raw Proteins (The "Fridge" Inventory)
        # Verified: Raw meats/fish focus
        {"id": "object-detection-f8udo/ingredients-v2", "version": 1},  # Verified: Raw Salmon, Pork, Scallops
        
        # 3. Produce (Fresh Inventory)
        {"id": "yolo-jpkho/combined-vegetables-fruits", "version": 1},  # Verified: Bulk Produce
    ]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMPORARY_RAW_IMAGES_FOLDER, exist_ok=True)

    temp_raw_dir = TEMPORARY_RAW_IMAGES_FOLDER
    purge_pantry_workspace([temp_raw_dir, output_dir])

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

    # ─── UNDERSAMPLING & TOP-UP LOGIC ───
    print(f"\n⚖️ Balancing Dataset: Aiming for {final_size} images per ingredient...")
    random.shuffle(all_data_pool)
    
    # Track which images belong to which class to handle undersampling/top-up
    class_map = {i: [] for i in range(len(ingredients))}
    for img_p, lbl_p in all_data_pool:
        try:
            with open(lbl_p, 'r') as f:
                lines = f.readlines()
                if not lines: continue
                class_id = int(lines[0].split()[0])
                # Only map if we haven't hit the undersampling cap (final_size)
                if class_id < len(ingredients) and len(class_map[class_id]) < final_size:
                    class_map[class_id].append((img_p, lbl_p))
        except: continue

    balanced_pool = []
    for class_id, pairs in class_map.items():
        current_count = len(pairs)
        balanced_pool.extend(pairs)
        
        # TOP-UP (Augmentation) if we are below final_size
        if 0 < current_count < final_size:
            num_needed = final_size - current_count
            print(f"🔄 Augmenting {ingredients[class_id]}: {current_count} -> {final_size}")
            for i in range(num_needed):
                src_img_p, src_lbl_p = random.choice(pairs)
                ext = os.path.splitext(src_img_p)[1]
                aug_img_p = src_img_p.replace(ext, f"_aug_{i}{ext}")
                aug_lbl_p = src_lbl_p.replace(".txt", f"_aug_{i}.txt")

                img = cv2.imread(src_img_p)
                if random.random() > 0.5: img = cv2.flip(img, 1) # Flip
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
                hsv[:,:,2] *= random.uniform(0.8, 1.2) # Brightness
                hsv[:,:,2][hsv[:,:,2] > 255] = 255
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                cv2.imwrite(aug_img_p, img)
                shutil.copy(src_lbl_p, aug_lbl_p)
                balanced_pool.append((aug_img_p, aug_lbl_p))

    all_data_pool = balanced_pool
    print(f"Total balanced image/label pairs: {len(all_data_pool)}")
    
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

    # 5.5. Dataset Balance Report
    print('─' * 55)
    print("\n--- 📊 Dataset Balance Report ---")
    print('─' * 55)
    label_counts = {name: 0 for name in ingredients}
    
    # Scan the labels folder in the final training set
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    for label_file in os.listdir(train_labels_dir):
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                if class_id < len(ingredients):
                    label_counts[ingredients[class_id]] += 1

    # Print results in a clean table
    print(f"{'Ingredient':<25} | {'Image Count':<12} | {'Status'}")
    print('─' * 55)
    for name, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        status = "✅ OK" if count >= 150 else "⚠️ LOW"
        if count == 0: status = "❌ MISSING"
        print(f"{name:<25} | {count:<12} | {status}")
    
    print('─' * 55)
    print()

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

    print('─────────────────────────────────────────────────────────────────────────')
    print('─────────────────────────────────────────────────────────────────────────')
    print(f"\n✅ DATASET READY: {output_dir}")
    print(f"Stats: Train({len(train)}) | Val({len(val)}) | Test({len(test)})")
    print(f"\nSUCCESS: Unified Dataset Created")
    print(f"Ready for Blackwell GPU training using data.yaml")
    print('─────────────────────────────────────────────────────────────────────────')
    print('─────────────────────────────────────────────────────────────────────────')

    # 8. Print Timing statistics
    elapsed = time.time() - imageload_start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print('─────────────────────────────────────────────────────────────────────────')
    print('─────────────────────────────────────────────────────────────────────────')
    print()
    print(f'Total Image Load and Merge runtime: {hours}h {minutes}m {seconds}s')
    print()
    print('─────────────────────────────────────────────────────────────────────────')
    print('─────────────────────────────────────────────────────────────────────────')


def load_pantry_images_from_open_images():

    purge_pantry_workspace([TEMPORARY_RAW_IMAGES_FOLDER, PANTRY_INGREDIENTS_IMAGES_FOLDER])

    # download_pantry_vision_dataset(
    #                                 ingredients=TEST_INGREDIENTS_SET, 
    #                                 pool_size=MAX_IMAGE_SAMPLES, 
    #                                 final_size=SAMPLES_PER_INGREDIENT,
    #                                 output_dir=PANTRY_INGREDIENTS_IMAGES_FOLDER
    #                             )


if __name__ == "__main__":
    # # Optional: Cleanup first
    # purge_dataset_folder(PANTRY_INGREDIENTS_IMAGES_FOLDER)
    
    # Execute the download and split pipeline
    load_pantry_images_from_open_images()
