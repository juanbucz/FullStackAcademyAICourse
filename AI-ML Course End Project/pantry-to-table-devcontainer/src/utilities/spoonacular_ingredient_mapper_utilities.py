# spoonacular_ingredient_mapper_utilities.py  
# ──────────────────────────────────────────────────────────────────────────────────────────
# Utility functions for mapping ingredients to Spoonacular Ingredient Vocabulary via LLM
# ──────────────────────────────────────────────────────────────────────────────────────────

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
class spoonacular_ingredient_mapper_utilities:
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
        self.system_prompt = """
            You are a specialized data mapping assistant for the Spoonacular Recipe API.

            Your Goal: Convert raw vision model labels into a standardized JSON dictionary.

            Rules:
            1. **Output Format**: Return ONLY a valid JSON object.
            2. **Key Preservation**: The Key must be EXACTLY the same string as provided in the input, preserving all underscores, spaces, and capitalization. This is critical for UI file resolution.
            3. **Value Standardization & Simplification**: 
            - The Value is the standardized Spoonacular name.
            - Use singular nouns.
            - Simplify varieties to base categories (e.g., 'Idaho potato' -> 'potato', 'red_bell_pepper' -> 'bell pepper').
            4. **Filter Unknowns**: Map non-food items to the string "unknown".

            ### FEW-SHOT EXAMPLES:
            Input: [red_onions, yellow onion, stainless_steel_knife]
            Output: {{"red_onions": "onion", "yellow onion": "onion", "stainless_steel_knife": "unknown"}}

            Input: [idaho_potatoes, russet potato, blue_ceramic_bowl]
            Output: {{"idaho_potatoes": "potato", "russet potato": "potato", "blue_ceramic_bowl": "unknown"}}

            Input: [fuji_apples, Granny Smith Apple, plastic_bag]
            Output: {{"fuji_apples": "apple", "Granny Smith Apple": "apple", "plastic_bag": "unknown"}}

            Input: [red_bell_pepper, Green Bell Pepper, battery]
            Output: {{"red_bell_pepper": "bell pepper", "Green Bell Pepper": "bell pepper", "battery": "unknown"}}

            Input: [ham_slices, Honey Ham, wooden_table]
            Output: {{"ham_slices": "ham", "Honey Ham": "ham", "wooden_table": "unknown"}}
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