# yolov8_utilities.py  
# ─────────────────────────────────────────────
# Utility functions for working with YoloV8[S] model
# ─────────────────────────────────────────────

import os
import cv2
import numpy as np
from typing import Tuple, Optional
from ultralytics import YOLO

# ─────────────────────────────────────────
# YoloV8 Model Wrapper API 
# ─────────────────────────────────────────
class YoloV8Utilities:
    """YoloV8 utility class for the Pantry App."""
    
    # ─────────────────────────────────────────
    # CONSTANTS
    # ─────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BEST_MODEL_WEIGHTS = os.path.join(BASE_DIR, 'yolov8m_best.pt')  
    
    def __init__(self, model_path=BEST_MODEL_WEIGHTS):
        
        # ─────────────────────────────────────────
        # Model Definition
        # ─────────────────────────────────────────
        # This loads best trained detection weights
        self.model = YOLO(model_path)

    def classify_ingredient(self, img)->tuple[str, float]:
    
        # 1. Convert Gradio's RGB to BGR for OpenCV/YOLO consistency
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get Prediction Results
        results = self.model.predict(source=img_bgr, device='0', conf=0.25)
        
        if len(results[0].boxes) > 0:

            # Grab the highest confidence detection
            top_box = results[0].boxes[0]
            class_id = int(top_box.cls)
            label = results[0].names[class_id]
            conf = float(top_box.conf)

            return label, conf
            
        return None, 0.0