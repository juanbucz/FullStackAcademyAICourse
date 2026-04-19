# spoonacular_ingredient_mapper_utilities.py  
# ──────────────────────────────────────────────────────────────────────────────────────────
# Utility functions for mapping ingredients to Spoonacular Ingredient Vocabulary via LLM
# ──────────────────────────────────────────────────────────────────────────────────────────
"""
    Predicted Ingredient to Spoonacular Ingredient Name Mapper

    To install ollama in VS Code DevContainer (via Terminal)

   sed -i 's|http://archive.ubuntu.com|https://mirrors.edge.kernel.org|g; s|http://security.ubuntu.com|https://mirrors.edge.kernel.org|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null
   apt-get update && apt-get install -y zstd

    apt-get update && apt-get install -y zstd
    curl -fsSL https://ollama.com/install.sh | sh

    To start up ollama client locally:

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
from dotenv import load_dotenv
from langchain_ollama import ChatOllama          # <--- ADD THIS
from langchain_openai import ChatOpenAI          # <--- ADD THIS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import JsonOutputParser


# ─────────────────────────────────────────
# Spoonacular LLM Wrapper API 
# ─────────────────────────────────────────
class SpoonacularIngredientMapperUtilities:
    """
        Predicted Ingredient to Spoonacular Ingredient Name Mapper

        To install ollama in VS Code DevContainer (via Terminal)

        apt-get update && apt-get install -y zstd
        curl -fsSL https://ollama.com/install.sh | sh

        To start up ollama client locally:

        $ ollama serve
        $ ollama pull qwen2.5:3b
    """


    # Load environment variables
    load_dotenv()

    
    def __init__(self):
        """ Constructor """        
        # ─────────────────────────────────────────
        # CONSTANTS
        # ─────────────────────────────────────────

        # Set Temperature to be strict/factual
        self.temperature = 0

        # ─────────────────────────────────────────
        # --- Initialize backends ---
        # ─────────────────────────────────────────

        self.ollama_model = 'qwen2.5:3b'
        self.ollama_client = ChatOllama(model=self.ollama_model, 
                                        temperature=self.temperature)

        # For Future Phase
        # llamacpp_server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

        # if llamacpp_server.startswith('localhost') or llamacpp_server.startswith('127.'):
        #     llamacpp_api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
        #     llamacpp_base_url = f'http://{llamacpp_server}/v1'
        # else:
        #     llamacpp_api_key = os.environ.get('PERDRIZET_API_KEY')
        #     llamacpp_base_url = f'https://{llamacpp_server}/v1'

        # llamacpp_client = ChatOpenAI(
        #     base_url=llamacpp_base_url,
        #     api_key=llamacpp_api_key,
        #     timeout=120.0,
        #     model='gpt-oss-20b',
        #     temperature=temperature
        # )

        # llamacpp_model = 'gpt-oss-20b'

        # ─────────────────────────────────────────
        # Prompt Definitions
        # ─────────────────────────────────────────
        #                 - Use singular nouns where appropriate.
        self.system_prompt = """
            You are a specialized data mapping assistant for the Spoonacular Recipe API.

            Your Goal: Convert raw ingredient labels into a standardized JSON dictionary for recipe searching.

            Rules:
            1. **Output Format**: Return ONLY a valid JSON object.
            2. **Key Preservation**: The Key must be EXACTLY the same string as provided in the input list. Do not add underscores, change case, or modify punctuation in the Key.
            3. **Value Standardization (Culinary Specificity)**: 
                - Keep essential culinary forms and cuts (e.g., 'chicken thigh', 'garlic powder', 'tomato paste', 'sirloin tips').
                - Strip marketing adjectives and regional varieties (e.g., 'Organic Fuji Apple' -> 'apple', 'Idaho potato' -> 'potato').
                - Do NOT over-generalize specific ingredients into broad categories (e.g., do NOT map 'linguine' to 'pasta' or 'basil' to 'herb').

            4. **Filter Unknowns**: Map clearly non-food items to the string "unknown".

            ### FEW-SHOT EXAMPLES:
            Input: ["Garlic Powder", "Fresh Basil", "Linguine pasta"]
            Output: {{"Garlic Powder": "garlic-powder", "Fresh Basil": "basil", "Linguine pasta": "linguine"}}

            Input: ["Chicken Thighs", "Boneless Chicken Breast", "Kitchen Scale"]
            Output: {{"Chicken Thighs": "chicken-thigh", "Boneless Chicken Breast": "chicken-breast", "Kitchen Scale": "unknown"}}

            Input: ["Russet Potatoes", "Granny Smith Apples", "Tomato Paste"]
            Output: {{"Russet Potatoes": "potato", "Granny Smith Apples": "apple", "Tomato Sauce": "tomato-sauce-or-pasta-sauce"}}

            Input: ["Red Bell Pepper", "Sharp Cheddar Cheese", "Cardboard Box"]
            Output: {{"Red Bell Pepper": "bell-pepper", "Sharp Cheddar Cheese": "cheddar-cheese", "Cardboard Box": "unknown"}}

            Input: ["Iceberg Lettuce", "Ground Beef", "Beets"]
            Output: {{"Iceberg Lettuce": "iceberg-lettuce", "Ground Beef": "fresh-ground-beef", "Beets": "beet"}}
            """


    def map_ingredients_to_spoonacular(self, detected_items: list[str], backend: str = 'Ollama') -> dict:
        """
        Standardizes YOLO detections and user entered custom items for the Spoonacular API using a Few-Shot prompt.
        Returns a dictionary mapping raw labels to standardized ingredients or 'unknown'.
        """

        llm = self.ollama_client if backend == 'Ollama' else self.llamacpp_client
        output_parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Detected objects from YOLO: {detected_items}")
        ])

        chain = prompt | llm | output_parser

        # Prepare input string (The parser handles the list effectively)
        try:
            # Returns: {"Ham": "ham", "tire": "unknown", "Idaho potato": "potato", ...}
            spoonaculared_dict = chain.invoke({"detected_items": detected_items})
            
            # Preserve ingredient name Case
            return spoonaculared_dict
            
        except Exception as e:
            print(f"Mapping Error: {e}")
            # Manual fallback to dictionary format
            return {i: "unknown" for i in detected_items}