"""
OCR ingredient label utilities.

Uses EasyOCR (torch-based) to share the same runtime as YOLO/ultralytics,
avoiding the MKL/OpenMP conflicts that PaddleOCR hit when co-resident with
PyTorch in the NGC container.

Install:
    pip install easyocr

 To start up ollama client locally:

        # For Image Classification
        $ ollama serve
        $ ollama pull qwen2.5:3b

        # For Image Label Classification
        $ ollama pull llama3    
        $ ollama pull mistral-small:22b

        ** Best Choice
        ** Provides best instruction-following and extraction logic specifically for messy OCR data
        $ ollama pull qwen2.5:14b        
"""

import os
import sys
import re

import numpy as np
from PIL import Image

import easyocr

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from utilities.spoonacular_utilities import SpoonacularUtilities as su


class OCRIngredientLabelUtilities:
    def __init__(self, ollama_model='qwen2.5:14b', use_gpu=True):
        """
        Initializes the OCR reader and Reasoning LLM.
        First call downloads EasyOCR model weights (~100MB) to ~/.EasyOCR/.
        """
        # 1. Initialize EasyOCR (torch-based, shares runtime with YOLO)
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        # 2. Reasoning LLM via Ollama
        self.llm = Ollama(model=ollama_model, temperature=0)

        # 3. LangChain few-shot glue
        self.chain = self.build_glue_chain()

        # Noise patterns
        self.weight_pattern = r'(\d+\s?(ml|g|oz|lb|kg|z|l))\b|net\s?wt'
        self.date_pattern = r'(\d{2}/\d{2})|(\d{4}-\d{2}-\d{2})|exp\b|best\s?by'

    # ---------- Noise filtering ----------

    def is_noise(self, text):
        clean_text = text.lower().strip()
        if re.search(self.weight_pattern, clean_text) or re.search(self.date_pattern, clean_text):
            return True
        return False

    # ---------- LLM glue chain ----------

    # def build_glue_chain(self):
    #     examples = [
    #         {'input': 'MCCORMICK | MONTREAL SPICE RUB | GRILL MATES', 'output': 'Montreal Spice Rub'},
    #         {'input': 'GREEN GIANT | SWEET PEAS | NET WT 12Z', 'output': 'Peas'},
    #         {'input': 'HEINZ | TOMATO KETCHUP | 500ML', 'output': 'Ketchup'},
    #         {'input': 'TYSON | BONELESS SKINLESS CHICKEN BREAST | FAMILY PACK', 'output': 'Chicken Breast'},
    #         {'input': 'Chicken Thighs', 'output': 'Chicken Thighs'},
    #         {'input': 'Black Beans', 'output': 'Black Beans'},
    #         {'input': 'ORGANIC WHOLE WHEAT PASTA | NON-GMO', 'output': 'Whole Wheat Pasta'},
    #     ]

    #     example_prompt = PromptTemplate(
    #         input_variables=['input', 'output'],
    #         template='Snippets: {input}\nIngredient: {output}',
    #     )

    #     return FewShotPromptTemplate(
    #         examples=examples,
    #         example_prompt=example_prompt,
    #         prefix='Task: Extract the primary food ingredient. Ignore brands, marketing, and weights.',
    #         suffix='Snippets: {input}\nIngredient:',
    #         input_variables=['input'],
    #     )

    def build_glue_chain(self):
        # 1. Define the format for each example
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Snippets: {input}\nIngredient: {output}"
        )

        # 2. Few-shot examples (Added 'Unknown' case)
        examples = [
            {"input": "MCCORMICK | MONTREAL SPICE RUB | GRILL MATES", "output": "Montreal Spice Rub"},
            {"input": "GREEN GIANT | SWEET PEAS | NET WT 12Z", "output": "Peas"},
            {"input": "HEINZ | TOMATO KETCHUP | 500ML", "output": "Ketchup"},
            {"input": "TYSON | BONELESS SKINLESS CHICKEN BREAST | FAMILY PACK", "output": "Chicken Breast"},
            {"input": "ORGANIC WHOLE WHEAT PASTA | NON-GMO", "output": "Whole Wheat Pasta"},
            {"input": "123456789 | XXXXX | @#$%^&", "output": "Unknown"} 
        ]

        # 3. Create the template with the 'Unknown' instruction
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="""Task: Extract the primary food ingredient. 
                        Rules:
                        - Output ONLY the name of the ingredient.
                        - If the text is garbage, random numbers, or unidentifiable, return 'Unknown'.
                        - Ignore brands, marketing, and weights.
                    """,
            suffix="Snippets: {input}\nIngredient:",
            input_variables=["input"],
        )

    # ---------- Spatial heuristic ----------

    def apply_spatial_heuristic(self, ocr_results):
        """
        Scores text blocks based on area, confidence, and internal noise filtering.
        EasyOCR returns a list of (box, text, confidence) tuples where
        box = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] in TL/TR/BR/BL order.
        """
        scored_data = []
        noise_words = {'net', 'wt', 'weight', 'serving', 'distributed', 'oz', 'ml', 'grams', 'expiration date'}

        if not ocr_results:
            return ''

        for item in ocr_results:
            box = item[0]         # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = item[1]
            confidence = item[2]

            if self.is_noise(text):
                continue

            width = box[1][0] - box[0][0]
            height = box[2][1] - box[1][1]
            area = max(width * height, 1.0)

            penalty = 2.5 if any(w in text.lower() for w in noise_words) else 1.0
            score = (area * confidence) / penalty
            scored_data.append((score, text))

        top_snippets = [item[1] for item in sorted(scored_data, reverse=True)[:5]]
        return ' | '.join(top_snippets)

    # ---------- Public entry point ----------

    def identify_ingredient(self, image_input):
        """
        Main entry point for converting image text to clean ingredient names.
        Accepts a PIL Image, numpy array, or file path.
        """
        # EasyOCR accepts file path, numpy array, or bytes — normalize to numpy.
        if isinstance(image_input, str):
            img_np = np.array(Image.open(image_input).convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            img_np = np.array(image_input.convert('RGB'))

        ocr_results = self.reader.readtext(img_np)

        context_string = self.apply_spatial_heuristic(ocr_results)
        if not context_string:
            return 'Unknown Ingredient'

        formatted_prompt = self.chain.format(input=context_string)
        clean_name = self.llm.invoke(formatted_prompt)

        print()
        print()
        print('─────' * 20)
        print(f'identify_ingredients results:')
        print(f'context_string: {context_string}')
        print(f'clean_name: {clean_name}')
        print(f'formatted_prompt: {formatted_prompt}')
        print('─────' * 20)
        print()
        print()

        # If the LLM still hallucinates or gives an empty string, default to Unknown
        if not clean_name or len(clean_name) < 2:
            return "Unknown"
        else:
            clean_name = clean_name.replace('Ingredient:', '').strip()
            return su.verify_ingredient(clean_name)


if __name__ == '__main__':
    # Standalone smoke test
    classifier = OCRIngredientLabelUtilities(ollama_model='llama3')
    if len(sys.argv) > 1:
        result = classifier.identify_ingredient(sys.argv[1])
        print(f'Identified: {result}')
    else:
        print('Pass an image path to test: python ocr_ingredient_label_utilities.py <image>')
