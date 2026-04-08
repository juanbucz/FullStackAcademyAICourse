"""Utility function for loading ingredient images for model building, tuning and testing.

    Open Images V7: Large collection of 9 million photos with pre-made labels for thousands of objects.

    FiftyOne: Python tool used to search, filter, and download only the specific images and labels you need.
"""

import os
import shutil
import random
#from roboflow import Roboflow
from sklearn.model_selection import train_test_split


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

def download_pantry_vision_dataset(ingredients, pool_size=MAX_IMAGE_SAMPLES, final_size=SAMPLES_PER_INGREDIENT, output_dir="pantry_data"):
    """
    Downloads and prepares a balanced, multi-split dataset for YOLOv8 object detection.

    Function Logic:
    * Downloads Data: Automatically fetches pool_size images per ingredient from the Open Images V7 dataset.
    * Enforces Quality: Filters for "clean" images containing exactly one verified object to reach final_size samples.
    * Balances Classes: Uses stratified sampling to ensure every ingredient has an equal number of images in every split.
    * Calculates Splits: Mathematically divides the data into 64% Train, 16% Validation, and 20% Test sets (via 80/20 nested splits).
    * Generates Labels: Automatically creates the .txt coordinate files and folder structure required for YOLOv8 training.

    Args:
        ingredients (list): List of strings representing the Open Images V7 class names to download.
        pool_size (int): The initial number of raw images to fetch per class before quality filtering.
        final_size (int): The target number of clean, single-item images to retain per class for the final dataset.
        output_dir (str): The root directory path where the split folders (train, val, test) will be created.

    Returns:
        None: Files are written directly to the local filesystem in the following structure:
            output_dir/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── val/
            │   ├── images/
            │   └── labels/
            └── test/
                ├── images/
                └── labels/
    """
    
    # ────────────────────────────────────────────────────
    # 1. DOWNLOAD (The Raw Pool)
    # Fetch the pool_size for each ingredient class
    # ────────────────────────────────────────────────────
    print(f"Downloading pool of {pool_size} images for {len(ingredients)} classes...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=ingredients,
        max_samples=len(ingredients) * pool_size, 
        only_matching=True
    )

    all_filepaths = []
    all_labels = []
    
    # ────────────────────────────────────────────────────
    # 2. FILTER (The "Quality" Step)
    # ────────────────────────────────────────────────────
    print(f"Filtering down to {final_size} 'clean' images per ingredient...")
    for label in ingredients:
        # 1. Use .exists() to ensure we only look at samples with detection data
        # 2. Match images containing ONLY one detection of the specific ingredient
        clean_view = dataset.exists("ground_truth").match(
            (F("ground_truth.detections").length() == 1) & 
            (F("ground_truth.detections.label").contains(label))
        )
        
        # Take exactly the final_size from the available clean images
        subset = clean_view.take(final_size)
        
        if len(subset) < final_size:
            print(f"Warning: Only found {len(subset)} clean images for {label}")

        for sample in subset:
            all_filepaths.append(sample.filepath)
            all_labels.append(label)

    # ────────────────────────────────────────────────────
    # 3. THE NESTED SPLIT (80/20 then 80/20)
    # ────────────────────────────────────────────────────
    train_val_imgs, test_imgs, train_val_labs, _ = train_test_split(
        all_filepaths, all_labels, test_size=0.20, stratify=all_labels, random_state=42
    )

    train_imgs, val_imgs, _, _ = train_test_split(
        train_val_imgs, train_val_labs, test_size=0.20, stratify=train_val_labs, random_state=42
    )

    # ────────────────────────────────────────────────────
    # 4. EXPORT TO YOLO FORMAT
    # ────────────────────────────────────────────────────
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    
    for split_name, file_list in splits.items():
        # Select samples from the original dataset that match our split lists
        split_view = dataset.select(dataset.match(F("filepath").is_in(file_list)).values("id"))
        
        split_view.export(
            export_dir=os.path.join(output_dir, split_name),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=ingredients
        )

    print(f"\nFinal Dataset: Train({len(train_imgs)}) | Val({len(val_imgs)}) | Test({len(test_imgs)})")

def purge_dataset_folder(folder_path):
    """
    Safely removes all images, labels, and subdirectories from the target folder.
    Ensures a clean slate for the next download and split run.
    """
    if os.path.exists(folder_path):
        try:
            print(f"Cleaning up: {folder_path}...")

            # rmtree removes the directory and all its contents (images/labels/yaml)
            shutil.rmtree(folder_path)
            
            # Recreate the base directory so it's ready for the next run
            os.makedirs(folder_path)
            print("Cleanup successful. Folder is now empty.")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            print("Tip: Ensure no other programs (like VS Code or a YOLO process) are using these files.")
    else:
        print(f"Folder not found: {folder_path}. Nothing to clean.")
        # Create it so the path is ready for the script
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
