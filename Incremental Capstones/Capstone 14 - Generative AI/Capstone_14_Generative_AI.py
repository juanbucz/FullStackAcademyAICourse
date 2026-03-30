"""LangChain basics demo

This demo demonstrates core LangChain concepts:
1. Chat models and LLM wrappers
2. Chat prompt templates
3. Output parsers
4. Basic chains

The demo uses Gradio to provide an interactive interface where you can:
- Try different prompt templates
- See structured output parsing
- Experiment with chained operations

Usage:
    python demos/langchain_patterns/langchain_demo.py
"""

import os
from typing import List
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()

# --- Configuration ---

temperature = 0.1

# --- Initialize backends ---

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


# --- Pydantic models for output parsing ---

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    key_phrases: List[str] = Field(description="Important phrases that support the sentiment")


class RecipeInfo(BaseModel):
    name: str = Field(description="Name of the dish")
    cuisine: str = Field(description="Type of cuisine")
    ingredients: List[str] = Field(description="List of main ingredients")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")


class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or profession")
    location: str = Field(description="City or country where person lives")


# --- Demo functions ---

def demo_simple_chain(text: str, backend: str) -> tuple[str, str]:
    """Demo 1: Simple chain with prompt template and string output."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains concepts concisely."),
        ("human", "Explain {topic} in 2-3 sentences."),
    ])
    
    # Create chain: prompt -> model -> string parser
    chain = prompt | llm | StrOutputParser()
    
    # Execute
    result = chain.invoke({"topic": text})
    
    explanation = f"""**Chain components:**
    1. Prompt template with system message and variable placeholder
    2. {backend} chat model
    3. StrOutputParser (extracts text from AIMessage)

    **Input:** topic = "{text}"
    """
    
    return result, explanation


def demo_sentiment_analysis(text: str, backend: str) -> tuple[str, str]:
    """Demo 2: Chain with structured output (JSON)."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Create output parser
    parser = JsonOutputParser(pydantic_object=SentimentAnalysis)
    
    # Create prompt with format instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a sentiment analysis expert. Analyze the sentiment of the given text.
        {format_instructions}"""),
        ("human", "{text}"),
    ])
    
    # Create chain
    chain = prompt | llm | parser
    
    try:
        # Execute
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Format output
        output = f"""**Sentiment:** {result['sentiment']}
        **Confidence:** {result['confidence']:.2%}
        **Key phrases:**
        {chr(10).join(f"- {phrase}" for phrase in result['key_phrases'])}"""
                
        explanation = f"""**Chain components:**
        1. Prompt template with format instructions
        2. {backend} chat model
        3. JsonOutputParser with Pydantic schema

        **Schema fields:**
        - sentiment (str): positive/negative/mixed
        - confidence (float): 0.0 to 1.0
        - key_phrases (list[str]): Supporting evidence
        """
        
        return output, explanation
    
    except Exception as e:
        return f"Error: {str(e)}", f"An error occurred during parsing. Try a different input or backend."


def demo_entity_extraction(text: str, backend: str, entity_type: str) -> tuple[str, str]:
    """Demo 3: Entity extraction with different schemas."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Choose schema based on entity type
    if entity_type == "Person":
        schema = PersonInfo

    elif entity_type == "Recipe":
        schema = RecipeInfo

    else:
        return "Invalid entity type", "Please select a valid entity type"
    
    parser = JsonOutputParser(pydantic_object=schema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract {entity_type} information from the text.
        {format_instructions}"""),
        ("human", "{text}"),
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "text": text,
            "entity_type": entity_type.lower(),
            "format_instructions": parser.get_format_instructions()
        })
        
        # Format output nicely
        output = "**Extracted information:**\n\n"

        for key, value in result.items():
            if isinstance(value, list):

                output += f"**{key.replace('_', ' ').title()}:**\n"
                output += "\n".join(f"- {item}" for item in value) + "\n\n"

            else:
                output += f"**{key.replace('_', ' ').title()}:** {value}\n"
        
        explanation = f"""**Chain components:**
        1. Prompt template with dynamic entity type
        2. {backend} chat model
        3. JsonOutputParser with {entity_type} schema

        **Selected schema:** {entity_type}
        **Fields:** {', '.join(schema.model_fields.keys())}
        """
        
        return output, explanation
    
    except Exception as e:
        return f"Error: {str(e)}", f"Make sure your text contains {entity_type.lower()} information."


def demo_few_shot(text: str, backend: str) -> tuple[str, str]:
    """Demo 4: Few-shot learning with prompt templates."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Few-shot prompt with examples
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a text style classifier. Classify the writing style as: technical, casual, formal, or creative."),
        ("human", "The efficacy of the proposed methodology was validated through rigorous experimental procedures."),
        ("ai", "technical"),
        ("human", "Hey! Just wanted to say this app is super cool and easy to use."),
        ("ai", "casual"),
        ("human", "We are pleased to inform you that your application has been approved."),
        ("ai", "formal"),
        ("human", "The moonlight danced across the waves like silver ribbons weaving through the night."),
        ("ai", "creative"),
        ("human", "{text}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"text": text})
    
    explanation = """**Chain components:**
    1. Prompt template with 4 few-shot examples
    2. Chat model (learns from examples)
    3. StrOutputParser

    **Few-shot examples:**
    - Technical: "efficacy", "methodology", "validated"
    - Casual: "Hey!", "super cool", informal language
    - Formal: "We are pleased", official tone
    - Creative: Metaphors, descriptive language

    The model learns the pattern from examples!
    """
    
    return result.strip(), explanation


# --- Build Gradio UI ---

## ======================================================
##
## String/Constants for Dropdowns
## To limit the scope of what the LLM has to process, I decided to use 
## fixed lists for my prompt parameters and options

BIKE_OPTIONS = ['Electric', 'Road', 'City', 'Hybrid', 'Mountain Bike']
DISCOUNT_OPTIONS = ['Regular Price', 'Customer Loyalty - 10%', 'Spring Sale - 15%', 'Last Years Models - 20%', 'Clearance 30%']
THEME_OPTIONS = ['Rugged', 'Sleek', 'Eco-friendly', 'Premium']
METHOD_OPTIONS = ['Zero-Shot', 'Few-Shot', 'Chain of Thought']

DEFAULT_BIKE = 'Select a Bike Type'
DEFAULT_DISCOUNT = 'Select A Discount'
DEFAULT_THEME = 'Select a Theme'
DEFAULT_METHOD = 'Select a Method'

# Define the "Empty" or "Default" state with CSS
# Default appearance of ad output block
DEFAULT_EMPTY_AD_HTML = """
<div style='background-color: #FFFDD0; 
            min-height: 300px; 
            padding: 20px; 
            border-radius: 8px; 
            border: 1px dashed #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-family: sans-serif;
            font-weight: bold;
            font-size: 28px;
            text-align: center;'>
    <i>Custom Generated  advertisement will appear here...</i>
</div>
"""

## ======================================================
##
## Function Definitions


# Removes instructions/prompt (DEFAULT_XXX) from dropdown choices
def clean_bike_list(evt: gr.SelectData):
    if evt.value == DEFAULT_BIKE:
        return gr.update()
    return gr.update(choices=BIKE_OPTIONS), ""

def clean_discount_list(evt: gr.SelectData):
    if evt.value == DEFAULT_DISCOUNT:
        return gr.update()
    return gr.update(choices=DISCOUNT_OPTIONS), ""

def clean_theme_list(evt: gr.SelectData):
    if evt.value == DEFAULT_THEME:
        return gr.update()    
    return gr.update(choices=THEME_OPTIONS), ""

def clean_method_list(evt: gr.SelectData):
    if evt.value == DEFAULT_METHOD:
        return gr.update()
    return gr.update(choices=METHOD_OPTIONS), ""

def create_zero_shot_ad_prompt(bike, discount_plan, ad_theme):
    return "", f'create_zero_shot_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}'

def create_few_shot_ad_prompt(bike, discount_plan, ad_theme):
    return "", f'create_few_shot_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}'

def create_chain_of_thought_ad_prompt(bike, discount_plan, ad_theme):
    return "", f'create_chain_of_thought_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}'

def generate_ad_campaign(bike, discount_plan, ad_theme, method):

    # Validate input parameters
    if bike == DEFAULT_BIKE:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Bike Type.</span>", DEFAULT_EMPTY_AD_HTML
    
    if discount_plan == DEFAULT_DISCOUNT:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Discount Plan.</span>", DEFAULT_EMPTY_AD_HTML
    
    if ad_theme == DEFAULT_THEME:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Ad Theme.</span>",  DEFAULT_EMPTY_AD_HTML
    
    if method == DEFAULT_METHOD:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Prompting Method.</span>",  DEFAULT_EMPTY_AD_HTML
    
    # Call Correct Prompt Creation Function

    match method:
        case 'Zero-Shot':
            create_zero_shot_ad_prompt(bike, discount_plan, ad_theme)
        case 'Few-Shot':
            create_few_shot_ad_prompt(bike, discount_plan, ad_theme)
        case 'Chain of Thought':
            create_chain_of_thought_ad_prompt(bike, discount_plan, ad_theme)
    
    print()
        


# Return a tuple of updates in the exact order they appear in 'outputs'
def reset_input_controls():
    return (
        gr.update(choices=[DEFAULT_BIKE] + BIKE_OPTIONS, value=DEFAULT_BIKE),
        gr.update(choices=[DEFAULT_DISCOUNT] + DISCOUNT_OPTIONS, value=DEFAULT_DISCOUNT),
        gr.update(choices=[DEFAULT_THEME] + THEME_OPTIONS, value=DEFAULT_THEME),
        gr.update(choices=[DEFAULT_METHOD] + METHOD_OPTIONS, value=DEFAULT_METHOD),
        "", # Clear out Error Label
        DEFAULT_EMPTY_AD_HTML # return default text/colour/size for ad output
    )

with gr.Blocks(title='Bicycle Advertisement Generator') as AddGenerator:
    
    gr.Markdown("# 🚲 AI Bicycle Ad Generator")

    gr.Markdown("""
    # LangChain basics demo
    
    Explore core LangChain concepts with interactive examples:
    - **Prompt templates** with variable substitution
    - **Structured output parsing** with Pydantic schemas
    - **Basic chains** composing multiple steps
    - **Few-shot learning** with example-driven prompts
    """)
    
    with gr.Row():

        # Product Details
        with gr.Column(variant='panel'):
            gr.Markdown('### 🚲 Bicycle Product Details')

            bike_type = gr.Dropdown(
                                        choices=[DEFAULT_BIKE] + BIKE_OPTIONS, 
                                        label='Bike Types',
                                        value=DEFAULT_BIKE,                 # Displayed first
                                        allow_custom_value=False,           # User CANNOT type custom text                               
                                        interactive=True
                                    )
            
            discount = gr.Dropdown(
                                        choices=[DEFAULT_DISCOUNT] + DISCOUNT_OPTIONS, 
                                        label='Available Discounts',
                                        value=DEFAULT_DISCOUNT,                 
                                        allow_custom_value=False,           
                                        interactive=True
                                    )

        # Prompting Strategy 
        with gr.Column(variant='panel'):
            gr.Markdown('### 🧠 Prompting Strategy')

            theme = gr.Dropdown(
                                    choices = [DEFAULT_THEME] + THEME_OPTIONS, 
                                    label='Marketing Theme',
                                    value=DEFAULT_THEME,                 
                                    allow_custom_value=False,           
                                    interactive=True
                                )        
            
            prompt_type = gr.Dropdown(
                                        choices = [DEFAULT_METHOD] + METHOD_OPTIONS, 
                                        label='Method',
                                        value=DEFAULT_METHOD,                 
                                        allow_custom_value=False,                                                   
                                        interactive=True
                                    )
   
    with gr.Row():
        # So buttons don't stack (default behaviour), create a minimal width column
        # Embed the buttons with a row defined in that column
        with gr.Column(scale=0, min_width=400): 
            with gr.Row():
                submit_btn = gr.Button("Generate Ad", variant="primary")
                reset_btn = gr.Button("Reset All Fields")
                
        # Left alignment by spacer, pushing columns to the left
        with gr.Column(scale=1):
            pass
    
    error_label = gr.Markdown(value="", visible=True)

    ad_output = gr.HTML(value=DEFAULT_EMPTY_AD_HTML,
                        label="Generated Advertisement")

    gr.Markdown("""
    ---
    
    ## Key takeaways
    
    1. **Prompt templates** make prompts reusable and maintainable
    2. **Output parsers** extract structured data reliably
    3. **Chains** compose multiple steps with the `|` operator
    4. **Pydantic schemas** ensure type-safe structured outputs
    5. **Few-shot examples** help models learn patterns
    
    **Next step:** Try Activity 4 to build your own LangChain chains!
    """)

    # Item is selected in DropDown list box; instructions prompt removed from list
    bike_type.select(fn=clean_bike_list, inputs=None, outputs=[bike_type, error_label])
    discount.select(fn=clean_discount_list, inputs=None, outputs=[discount, error_label])
    theme.select(fn=clean_theme_list, inputs=None, outputs=[theme, error_label])
    prompt_type.select(fn=clean_method_list, inputs=None, outputs=[prompt_type, error_label])        

    submit_btn.click(
        fn=generate_ad_campaign,
        inputs=[bike_type, discount, theme, prompt_type],
        outputs=[error_label, ad_output]
    )      

    reset_btn.click(
        fn=reset_input_controls,
        inputs=None,
        outputs=[bike_type, discount, theme, prompt_type, error_label, ad_output]
    )     


# Launch the Gradio app
if __name__ == '__main__':
    AddGenerator.launch()
