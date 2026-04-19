
"""

    Issues installing paddleOCR in devcontainer
    To fix:

    For CPU:
        pip install --ignore-installed PyYAML && pip install paddleocr paddlepaddle 

    For GPU:
        pip uninstall -y paddlepaddle paddlepaddle-gpu
        pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

        [To Check GPU Installation]
        export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
        python -c "import paddle; paddle.utils.run_check()"

        THIS DOES NOT WORK!! ISSUES WITH LIBRARY COLLISION AND DEPENDENCIES 
        - Paddle and PyTorch's bundled MKL runtimes conflict, crashing paddle at inference.
        - NGC's global LD_LIBRARY_PATH leaks into subprocesses, so isolation didn't help.

        Long term fix is to create a dedicated Paddle devcontainer and host it in a webservice

"""

import numpy as np
import re
import os

from PIL import Image
import numpy as np

# force paddle's MKL to load before torch's
import paddle  # noqa: F401
from paddleocr import PaddleOCR  # noqa: F401

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


class OCRIngredientLabelUtilities:
    def __init__(self, ollama_model='llama3', use_gpu=False):
        """
        Initializes the OCR engine and the Reasoning LLM.
        """
        # 1. Initialize PaddleOCR
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            device='gpu' if use_gpu else 'cpu'
        )
        
        # 2. Initialize the Reasoning LLM via Ollama
        self.llm = Ollama(model=ollama_model, temperature=0)
        
        # 3. Setup the LangChain Few-Shot Glue Layer
        self.chain = self.build_glue_chain()
        
        # Weights/Volumes: 500ml, 12oz, 1kg, Net Wt, etc.
        self.weight_pattern = r'(\d+\s?(ml|g|oz|lb|kg|z|l))\b|net\s?wt'
        
        # Dates: 10/26, 2026-04-18, exp, best by, etc.
        self.date_pattern = r'(\d{2}/\d{2})|(\d{4}-\d{2}-\d{2})|exp\b|best\s?by'
        
    def is_noise(self, text):
        """Returns True if text matches patterns for weights, volumes, or dates."""
        clean_text = text.lower().strip()
        
        if re.search(self.weight_pattern, clean_text) or re.search(self.date_pattern, clean_text):
            return True
            
        return False        

    def build_glue_chain(self):
        """
        Configures the semantic reasoning logic with few-shot examples.
        """
        examples = [
            {'input': 'MCCORMICK | MONTREAL SPICE RUB | GRILL MATES', 'output': 'Montreal Spice Rub'},
            {'input': 'GREEN GIANT | SWEET PEAS | NET WT 12Z', 'output': 'Peas'},
            {'input': 'HEINZ | TOMATO KETCHUP | 500ML', 'output': 'Ketchup'},
            {'input': 'TYSON | BONELESS SKINLESS CHICKEN BREAST | FAMILY PACK', 'output': 'Chicken Breast'},
            {'input': 'Chicken Thighs', 'output': 'Chicken Thighs'},
            {'input': 'Black Beans', 'output': 'Black Beans'},
            {'input': 'ORGANIC WHOLE WHEAT PASTA | NON-GMO', 'output': 'Whole Wheat Pasta'}
        ]

        example_prompt = PromptTemplate(
            input_variables=['input', 'output'],
            template='Snippets: {input}\nIngredient: {output}'
        )

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix='Task: Extract the primary food ingredient. Ignore brands, marketing, and weights.',
            suffix='Snippets: {input}\nIngredient:',
            input_variables=['input']
        )

    def apply_spatial_heuristic(self, ocr_results):
        """
        Scores text blocks based on area, confidence, and internal noise filtering.
        """
        scored_data = []
        noise_words = {'net', 'wt', 'weight', 'serving', 'distributed', 'oz', 'ml', 'grams', 'expiration date'}

        if not ocr_results or not ocr_results[0]:
            return ''

        for line in ocr_results[0]:
            box = line[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            text = line[1][0]
            confidence = line[1][1]

            # --- INTEGRATED NOISE FILTER ---
            if self.is_noise(text):
                continue

            # Calculate Area
            width = box[1][0] - box[0][0]
            height = box[2][1] - box[1][1]
            area = width * height
            
            # Apply Penalty for meta-data words
            penalty = 2.5 if any(w in text.lower() for w in noise_words) else 1.0
            
            # Final Score Calculation
            score = (area * confidence) / penalty
            scored_data.append((score, text))

        # Sort by score and take top 5 snippets
        top_snippets = [item[1] for item in sorted(scored_data, reverse=True)[:5]]
        return ' | '.join(top_snippets)

    def identify_ingredient(self, image_input):
        """
        Main entry point for converting image text to clean ingredient names.
        """
        # Load/Convert Image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            if isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input).convert('RGB')
            else:
                img = image_input.convert('RGB')
            
        img_np = np.array(img)

        # Step 1: Physical Scan (PaddleOCR)
        raw_ocr_results = self.ocr_engine.ocr(img_np)

        # Step 2: Spatial Filtering (Heuristic) with is_noise call
        context_string = self.apply_spatial_heuristic(raw_ocr_results)
        
        if not context_string:
            return 'Unknown Ingredient'

        # Step 3: Semantic Reasoning (LLM Glue)
        formatted_prompt = self.chain.format(input=context_string)
        clean_name = self.llm.invoke(formatted_prompt)

        return clean_name.strip()


if __name__ == '__main__':
    intel = OCRIntelligence(ollama_model='llama3')

# --- Example Usage ---
if __name__ == '__main__':
    # Test block for standalone verification
    intel = OCRIntelligence(ollama_model='llama3')
    # result = intel.identify_ingredient("test_product.jpg")
    # print(f"Identified: {result}")