"""Essentials and Applications of Generative AI 

This Application allows users to:
1. Select Desired Prompt Technique
2. Specify the following:
    - Bike Style
    - Discount Plan
    - Ad Theme

The demo uses Gradio to provide an interactive interface where you can:
- Try different prompt templates
- See structured output parsing

Usage:
    python llms-demo-johnb/Capstones/Capstone 14/Capstone_14_Generative_AI.py 
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

# --- Demo functions ---

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

BIKE_OPTIONS = ['Electric', 'Racing', 'Road', 'City', 'Hybrid', 'Mountain']
DISCOUNT_OPTIONS = ['Customer Loyalty - 10%', 'Spring Sale - 15%', 'Last Years Models - 20%', 'Clearance 30%']
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
    
    #return "", f"<span style='color: green; font-weight: bold; font-size: 22px;'>⚠️ create_zero_shot_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}</span>"
    # Using ChatPromptTemplate with a System Message enforces better "Zero-Shot" behavior
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional marketing expert. Output ONLY valid HTML and inline CSS. Every tag MUST have a style attribute."),
        ("human", """Create a detailed advertisement for a {bike} bike.
        Theme: {ad_theme}
        Discount: {discount_plan}
        
        Requirements:
        1. **Outer Wrapper**: 
           - Use `<div style="background-color: #FFFDD0; padding: 25px; max-width: 600px; margin: 0 auto; border: 2px solid #0056b3; border-radius: 10px; font-family: Arial, sans-serif;">`

        2. **The Header**: 
           - Use `<div style="background-color: #0056b3; color: #FFFFFF !important; font-size: 28px; font-weight: bold; padding: 15px; text-align: center; border-radius: 5px; margin-bottom: 20px;">BikeEase: {bike} Adventure</div>`

        3. **The Image**: 
           - Use `<img src="https://loremflickr.com/300/200/{bike},bicycle" style="float: left; width: 220px; margin: 0 20px 10px 0; border-radius: 5px;">`

        4. **Body & Features**:
           - Write 4-5 detailed sentences of marketing copy. Style: `style="color: #000000; font-size: 16px; line-height: 1.5;"`
           - Directly under the sentences, add a `<ul>` list of 4 key features.
           - Discount style: `<span style="background-color: yellow; font-weight: bold; padding: 2px 5px; border: 1px solid black;">{discount_plan}</span>`
           - Features container style: `style="clear: both; margin-top: 20px; color: #000000;"`

        5. **The Button**: 
           - Use `<a href="#" style="display: block; width: 250px; margin: 20px auto 0; background-color: #0056b3 !important; color: #FFFFFF !important; font-size: 20px; font-weight: bold; padding: 15px; text-align: center; text-decoration: none; border-radius: 5px;">BikeEase - Shop Now - {discount_plan} Off</a>`
        """)
    ])
    
    # The Chain remains the same
    chain = prompt | ollama_client | StrOutputParser()
    
    # ErrorLabel, <HTML> Ad Content
    return "", chain.invoke({
        "bike": bike,
        "ad_theme": ad_theme,
        "discount_plan": discount_plan
    })    

def create_few_shot_ad_prompt(bike, discount_plan, ad_theme):
    return "", f"<span style='color: green; font-weight: bold; font-size: 22px;'>⚠️ create_few_shot_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}</span>"

def create_chain_of_thought_ad_prompt(bike, discount_plan, ad_theme):
    return "", f"<span style='color: green; font-weight: bold; font-size: 22px;'>⚠️ create_chain_of_thought_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}</span>"

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

    try:
        match method:
            case 'Zero-Shot':
                return create_zero_shot_ad_prompt(bike, discount_plan, ad_theme)
            case 'Few-Shot':
                return create_few_shot_ad_prompt(bike, discount_plan, ad_theme)
            case 'Chain of Thought':
                return create_chain_of_thought_ad_prompt(bike, discount_plan, ad_theme)

    except Exception as e:
        # Show exception in error label
        error_msg = f"<div style='color: red; font-weight: bold;'>⚠️ System Error: {str(e)}</div>"
        return error_msg, DEFAULT_EMPTY_AD_HTML                
    
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
    # This Application allows users to:
        1. Select Desired Prompt Technique
        2. Specify the following:
            - Bike Style
            - Discount Plan
            - Ad Theme

        The demo uses Gradio to provide an interactive interface where you can:
        - Try different prompt templates
        - See structured output parsing
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
