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

    # ─────────────────────────────────────────
    # CONSTANTS
    # ─────────────────────────────────────────

    # Set Temperature to be strict/factual
    temperature = 0

    # ─────────────────────────────────────────
    # --- Initialize backends ---
    # ─────────────────────────────────────────

    ollama_model = 'qwen2.5:3b'
    ollama_client = ChatOllama(model=ollama_model, temperature=temperature)

    llamacpp_server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

    if llamacpp_server.startswith('localhost') or llamacpp_server.startswith('127.'):
        llamacpp_api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
        llamacpp_base_url = f'http://{llamacpp_server}/v1'
    else:
        llamacpp_api_key = os.environ.get('PERDRIZET_API_KEY')
        llamacpp_base_url = f'https://{llamacpp_server}/v1'

    llamacpp_client = ChatOpenAI(
        base_url=llamacpp_base_url,
        api_key=llamacpp_api_key,
        timeout=120.0,
        model='gpt-oss-20b',
        temperature=temperature
    )

    llamacpp_model = 'gpt-oss-20b'

    
    def __init__(self):
        """ Constructor """        
        # ─────────────────────────────────────────
        # Model Definition
        # ─────────────────────────────────────────

    def map_ingredients_to_spoonacular(self, detected_items: list[str], backend: str) -> list[str]:
        """
        Standardizes YOLO detections for the Spoonacular API using a Few-Shot prompt.
        """
        # Select backend (defaulting to the client initialized above)
        llm = self.ollama_client if backend == 'Ollama' else self.llamacpp_client
        output_parser = CommaSeparatedListOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """
            
            You are a specialized data mapping assistant for the Spoonacular Recipe API.
            
            Your Goal: Convert raw vision model labels into standardized, searchable Spoonacular ingredient names.
            
            Rules:
            1. **Standardize**: Use the base, singular noun (e.g., 'onions' -> 'onion').
            2. **Filter**: Completely remove non-food items (e.g., 'bowl', 'knife', 'fork', 'person').
            3. **Normalize**: Replace underscores with spaces and remove descriptors (e.g., 'fuji_apple' -> 'apple').
            4. **Format**: Output ONLY a comma-separated list of strings. No conversation.
            
            ### EXAMPLES:
            Input: [red_onions, stainless_steel_knife, sharp_cheddar_cheese, plastic_bag, gala_apples]
            Output: onion, cheddar cheese, apple

            Input: [organic_whole_milk, blue_ceramic_bowl, large_brown_eggs, carton]
            Output: milk, egg
            """),
            ("human", "Detected objects from YOLO: {detected_items}")
        ])

        chain = prompt | llm | output_parser

        # Prepare input string
        detected_str = ", ".join(detected_items)
        
        try:
            # Standardize via LLM
            standardized_list = chain.invoke({"detected_items": detected_str})
            
            # Clean up: strip whitespace and ensure lowercase
            return [item.strip().lower() for item in standardized_list if item.strip()]
            
        except Exception as e:
            print(f"Mapping Error: {e}")
            # Manual fallback logic
            return [i.replace('_', ' ').lower() for i in detected_items if i not in ['potato', 'apple', 'grapefruit']]