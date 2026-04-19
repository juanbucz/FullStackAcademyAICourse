"""
OCR ingredient label utilities.

PaddleOCR runs in a subprocess (ocr_worker.py) to avoid MKL conflicts with
PyTorch/ultralytics when both are loaded in the same Python process.

Install notes:
    For CPU:
        pip install --ignore-installed PyYAML && pip install paddleocr paddlepaddle

    For GPU (currently unused — Blackwell + NGC PyTorch coexistence issues):
        pip uninstall -y paddlepaddle paddlepaddle-gpu
        pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu129/


        THIS DOES NOT WORK!! ISSUES WITH LIBRARY COLLISION AND DEPENDENCIES 
        - Paddle and PyTorch's bundled MKL runtimes conflict, crashing paddle at inference.
        - NGC's global LD_LIBRARY_PATH leaks into subprocesses, so isolation didn't help.

        Long term fix is to create a dedicated Paddle devcontainer and host it in a webservice        
"""

import os
import sys
import re
import json
import tempfile
import subprocess

import numpy as np
from PIL import Image

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


# Path to the subprocess worker script (sits alongside this file)
_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_worker.py')


class OCRIngredientLabelUtilities:
    def __init__(self, ollama_model='llama3', use_gpu=False):
        """
        Initializes the Reasoning LLM. PaddleOCR runs out-of-process.
        `use_gpu` is accepted for API compatibility but currently has no effect
        (subprocess worker is pinned to CPU to keep the environment simple).
        """
        if not os.path.exists(_WORKER_PATH):
            raise FileNotFoundError(f'OCR worker script not found at {_WORKER_PATH}')

        # Reasoning LLM via Ollama
        self.llm = Ollama(model=ollama_model, temperature=0)

        # LangChain few-shot glue
        self.chain = self.build_glue_chain()

        # Noise patterns
        self.weight_pattern = r'(\d+\s?(ml|g|oz|lb|kg|z|l))\b|net\s?wt'
        self.date_pattern = r'(\d{2}/\d{2})|(\d{4}-\d{2}-\d{2})|exp\b|best\s?by'

    # ---------- OCR via subprocess ----------

    def _run_ocr_subprocess(self, image_path: str) -> list:
        """
        Invoke ocr_worker.py as a fresh Python process. Returns a list of
        {'text': str, 'confidence': float, 'box': [[x,y], ...]}.
        """
        result = subprocess.run(
            [sys.executable, _WORKER_PATH, image_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'OCR worker failed (exit {result.returncode}):\n{result.stderr}'
            )
        return json.loads(result.stdout)

    # ---------- Noise filtering ----------

    def is_noise(self, text):
        clean_text = text.lower().strip()
        if re.search(self.weight_pattern, clean_text) or re.search(self.date_pattern, clean_text):
            return True
        return False

    # ---------- LLM glue chain ----------

    def build_glue_chain(self):
        examples = [
            {'input': 'MCCORMICK | MONTREAL SPICE RUB | GRILL MATES', 'output': 'Montreal Spice Rub'},
            {'input': 'GREEN GIANT | SWEET PEAS | NET WT 12Z', 'output': 'Peas'},
            {'input': 'HEINZ | TOMATO KETCHUP | 500ML', 'output': 'Ketchup'},
            {'input': 'TYSON | BONELESS SKINLESS CHICKEN BREAST | FAMILY PACK', 'output': 'Chicken Breast'},
            {'input': 'Chicken Thighs', 'output': 'Chicken Thighs'},
            {'input': 'Black Beans', 'output': 'Black Beans'},
            {'input': 'ORGANIC WHOLE WHEAT PASTA | NON-GMO', 'output': 'Whole Wheat Pasta'},
        ]

        example_prompt = PromptTemplate(
            input_variables=['input', 'output'],
            template='Snippets: {input}\nIngredient: {output}',
        )

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix='Task: Extract the primary food ingredient. Ignore brands, marketing, and weights.',
            suffix='Snippets: {input}\nIngredient:',
            input_variables=['input'],
        )

    # ---------- Spatial heuristic ----------

    def apply_spatial_heuristic(self, ocr_results):
        """
        Scores text blocks based on area, confidence, and internal noise filtering.
        Operates on the flat list-of-dicts produced by the subprocess worker.
        """
        scored_data = []
        noise_words = {'net', 'wt', 'weight', 'serving', 'distributed', 'oz', 'ml', 'grams', 'expiration date'}

        if not ocr_results:
            return ''

        for item in ocr_results:
            text = item['text']
            confidence = item['confidence']
            box = item['box']  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]

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
        # Normalize input to a PIL Image
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        # Save to a temp file the worker can read (subprocess needs a path)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img.save(tmp_path)
            ocr_results = self._run_ocr_subprocess(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        context_string = self.apply_spatial_heuristic(ocr_results)
        if not context_string:
            return 'Unknown Ingredient'

        formatted_prompt = self.chain.format(input=context_string)
        clean_name = self.llm.invoke(formatted_prompt)
        return clean_name.strip()


if __name__ == '__main__':
    # Standalone smoke test
    classifier = OCRIngredientLabelUtilities(ollama_model='llama3')
    if len(sys.argv) > 1:
        result = classifier.identify_ingredient(sys.argv[1])
        print(f'Identified: {result}')
    else:
        print('Pass an image path to test: python ocr_ingredient_label_utilities.py <image>')
