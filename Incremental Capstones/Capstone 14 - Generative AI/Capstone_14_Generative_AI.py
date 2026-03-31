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

temperature = 0.5

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

DEFAULT_REASONING_CHAIN='Chain of Thought Reasoning Steps Will Appear Here...'


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

def create_zero_shot_ad_prompt(bike: str, discount_plan: str, ad_theme: str) -> tuple[str, str, str]:
    
    #return "", f"<span style='color: green; font-weight: bold; font-size: 22px;'>⚠️ create_zero_shot_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}</span>"
    # Using ChatPromptTemplate with a System Message enforces better "Zero-Shot" behavior
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional marketing expert. Output ONLY valid HTML and inline CSS. Every tag MUST have a style attribute."),
        ("human", """Create a detailed advertisement for a {bike} bike.
        Theme: {ad_theme}
        Discount: {discount_plan}
        
        Requirements:
        1. **Format**: Output ONLY raw HTML. Do not use markdown (```) or the word 'html'.
         
        2. **Thematic Palette**: 
           - You MUST select a background-color that represents {ad_theme} (e.g., Earthy for Rugged, Charcoal/Dark for Sleek, Green for Eco-friendly).
           - **CRITICAL**: Set `color: #FFFFFF !important;` for the Outer Wrapper to ensure all text is white and readable.

        3. **Outer Wrapper**: 
           - Use a <div> with the background-color from Rule 2 and white text.
           - Style: `padding: 25px; max-width: 600px; margin: 0 auto; border: 2px solid #333; border-radius: 10px; font-family: Arial, sans-serif; overflow: hidden;`

        4. **The Header**: 
           - Use `<div style="background-color: #FFFFFF; color: #000000 !important; font-size: 28px; font-weight: bold; padding: 15px; text-align: center; border-radius: 5px; margin-bottom: 20px; border: 1px solid #000;">BikeEase: {bike} Adventure</div>`
         
        5. **The Image**: 
           - Use `<img src="https://loremflickr.com/300/200/{bike},bicycle" style="float: left; width: 220px; margin: 0 20px 10px 0; border-radius: 5px; border: 1px solid #FFF;">`         

        6. **Body & Features**:
           - Next to the image, write 4-5 marketing sentences strictly using the vocabulary and brand voice of the {ad_theme} theme. Style: `style="font-size: 16px; line-height: 1.5; color: #FFFFFF;"`
           - Underneath the image and sentences, add a `<ul>` list of 4 key features tailored to the {ad_theme} nature. **IMPORTANT**: Use `style="clear: both; padding-top: 15px; color: #FFFFFF;"` for the list.
           - Discount: `<span style="background-color: yellow; color: #000000 !important; font-weight: bold; padding: 2px 5px; border: 1px solid black;">{discount_plan}</span>`

        7. **The Button**: 
           - Use `<a href="#" style="display: block; width: 250px; margin: 20px auto 0; background-color: #FFFFFF !important; color: #000000 !important; font-size: 20px; font-weight: bold; padding: 15px; text-align: center; text-decoration: none; border-radius: 5px; border: 2px solid #000;">Shop Now - {discount_plan} Off</a>`
        """)
    ])
    
    # The Chain remains the same
    chain = prompt | ollama_client | StrOutputParser()

    # Strip any leading/trailing backticks or 'html' labels before returning
    result = chain.invoke({
        "bike": bike, 
        "ad_theme": ad_theme, 
        "discount_plan": discount_plan
    })

    # Cleaning backticks
    clean_result = result.replace("```html", "").replace("```", "").strip()
    
    # ErrorLabel, <HTML> Ad Content
    return "", clean_result, DEFAULT_REASONING_CHAIN

def create_few_shot_ad_prompt(bike: str, discount_plan: str, ad_theme: str) -> tuple[str, str, str]:

    #return "", f"<span style='color: green; font-weight: bold; font-size: 22px;'>⚠️ create_few_shot_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}</span>"

    # Few-shot prompt with diverse examples (Rugged, Sleek, and Premium)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a marketing expert. Output ONLY raw HTML. Do not use markdown code blocks or backticks."),
        
        # Rugged (Dark/Earth Theme)
        ("human", "Create a detailed advertisement for a Mountain bike. Theme: Rugged. Discount: 15% Off."),
        ("ai", """<div style="background-color: #4B3621; color: #FFFFFF; padding: 25px; max-width: 600px; margin: 0 auto; border: 2px solid #333; border-radius: 10px; font-family: Arial, sans-serif;">
            <div style="background-color: #FFFFFF; color: #000000; font-size: 28px; font-weight: bold; padding: 15px; text-align: center; border-radius: 5px; margin-bottom: 20px;">BikeEase: Mountain Adventure</div>
            <img src="https://loremflickr.com/300/200/Mountain,bicycle" style="float: left; width: 220px; margin: 0 20px 10px 0; border-radius: 5px;">
            <div style="font-size: 16px; line-height: 1.5; color: #FFFFFF; clear: both;">
                <p>Conquer the toughest trails with the **unyielding** performance of our Mountain bike. Built with a **heavy-duty** frame for **all-terrain** dominance, this machine is truly **rugged**.</p>
                <ul>
                    <li>Reinforced Steel Frame</li>
                    <li>Aggressive Knobby Tires</li>
                    <li>Dual-Suspension System</li>
                    <li>15% Off Discount Applied</li>
                </ul>
            </div>
            <a href="#" style="display: block; width: 250px; margin: 20px auto 0; background-color: #0056b3; color: #ffffff; font-size: 20px; font-weight: bold; padding: 15px; text-align: center; text-decoration: none; border-radius: 5px;">Shop Now</a>
        </div>"""),

        # Sleek (Light/Modern Theme)
        ("human", "Create a detailed advertisement for a Road bike. Theme: Sleek. Discount: 10% Off."),
        ("ai", """<div style="background-color: #F0F0F0; color: #333333; padding: 25px; max-width: 600px; margin: 0 auto; border: 2px solid #0056b3; border-radius: 10px; font-family: Arial, sans-serif;">
            <div style="background-color: #0056b3; color: #FFFFFF; font-size: 28px; font-weight: bold; padding: 15px; text-align: center; border-radius: 5px; margin-bottom: 20px;">BikeEase: Road Adventure</div>
            <img src="https://loremflickr.com/300/200/Road,bicycle" style="float: left; width: 220px; margin: 0 20px 10px 0; border-radius: 5px;">
            <div style="font-size: 16px; line-height: 1.5; color: #333333; clear: both;">
                <p>Experience **precision** engineering and **minimalist** design. Our Road bike offers an **aerodynamic** profile for a **smooth**, high-speed commute.</p>
                <ul>
                    <li>Ultra-light Carbon Frame</li>
                    <li>Integrated Cable Routing</li>
                    <li>High-Pressure Slick Tires</li>
                    <li>10% Off Limited Offer</li>
                </ul>
            </div>
            <a href="#" style="display: block; width: 250px; margin: 20px auto 0; background-color: #0056b3; color: #ffffff; font-size: 20px; font-weight: bold; padding: 15px; text-align: center; text-decoration: none; border-radius: 5px;">Shop Now</a>
        </div>"""),

        # Premium (Luxury/Black & Gold Theme)
        ("human", "Create a detailed advertisement for a Cruiser bike. Theme: Premium. Discount: 5% Off."),
        ("ai", """<div style="background-color: #000000; color: #D4AF37; padding: 25px; max-width: 600px; margin: 0 auto; border: 2px solid #D4AF37; border-radius: 10px; font-family: 'Times New Roman', serif;">
            <div style="background-color: #D4AF37; color: #000000; font-size: 28px; font-weight: bold; padding: 15px; text-align: center; border-radius: 5px; margin-bottom: 20px;">BikeEase: Exclusive Cruiser</div>
            <img src="https://loremflickr.com/300/200/Cruiser,bicycle" style="float: left; width: 220px; margin: 0 20px 10px 0; border-radius: 5px; border: 1px solid #D4AF37;">
            <div style="font-size: 16px; line-height: 1.5; color: #D4AF37; clear: both;">
                <p>Indulge in **unrivaled** luxury with our **bespoke** Cruiser. Every detail is meticulously crafted for the **sophisticated** rider who demands **elegant** aesthetics and superior comfort.</p>
                <ul>
                    <li>Hand-Stitched Leather Accents</li>
                    <li>Polished Chrome Finish</li>
                    <li>Silent-Drive Technology</li>
                    <li>Exclusive 5% Member Invitation</li>
                </ul>
            </div>
            <a href="#" style="display: block; width: 250px; margin: 20px auto 0; background-color: #D4AF37; color: #000000; font-size: 20px; font-weight: bold; padding: 15px; text-align: center; text-decoration: none; border-radius: 5px;">Reserve Yours</a>
        </div>"""),

        # The Actual Request
        ("human", "Create a detailed advertisement for a {bike} bike. Theme: {ad_theme}. Discount: {discount_plan}."),
    ])
    
    chain = prompt | ollama_client | StrOutputParser()
    
    result = chain.invoke({
        "bike": bike, 
        "ad_theme": ad_theme, 
        "discount_plan": discount_plan
    })

    # Cleaning backticks
    clean_result = result.replace("```html", "").replace(": HTML GENERATION", "").replace("```", "").strip()
    
    return "", clean_result, DEFAULT_REASONING_CHAIN


def create_chain_of_thought_ad_prompt(bike: str, discount_plan: str, ad_theme: str) -> tuple[str, str, str]:
    
    # return "", f"<span style='color: green; font-weight: bold; font-size: 22px;'>⚠️ create_chain_of_thought_ad_prompt -- bike: {bike},  discount_plan: {discount_plan}, ad_theme: {ad_theme}</span>"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Strict Protocol: You are a two-stage generator. You are FORBIDDEN from generating HTML until you have written '### STEP 1: DESIGN REASONING'. You must output BOTH steps."),
        ("human", """[CRITICAL: DO NOT SKIP STEP 1]
        
        ### TASK: Create a {ad_theme} advertisement for a {bike} bike.
        
        ### REQUIRED OUTPUT STRUCTURE:
        You must strictly follow this sequence:
        1. Write the header: '### STEP 1: DESIGN REASONING'
        2. Provide your analysis of the {ad_theme} theme and color palette.
        3. Write the header: '### STEP 2: HTML GENERATION'
        4. Provide the raw HTML code.

        --- 
        ### STEP 2 HTML RULES (DO NOT USE MARKDOWN BACKTICKS):
        1. **Outer Wrapper**: <div> with {ad_theme} background-color and `color: #FFFFFF !important;`. 
           Style: `padding: 25px; max-width: 600px; margin: 0 auto; border: 2px solid #333; border-radius: 10px; font-family: Arial, sans-serif; overflow: hidden;`
        
        2. **The Header**: 
           `<div style="background-color: #FFFFFF; color: #000000 !important; font-size: 28px; font-weight: bold; padding: 15px; text-align: center; border-radius: 5px; margin-bottom: 20px; border: 1px solid #000;">BikeEase: {bike} Adventure</div>`
        
        3. **The Image**: 
           `<img src="https://loremflickr.com/300/200/{bike},bicycle" style="float: left; width: 220px; margin: 0 20px 10px 0; border-radius: 5px; border: 1px solid #FFF;">`
        
        4. **Body & Features**: 
           - Write 4-5 marketing sentences in {ad_theme} voice. Style: `color: #FFFFFF; font-size: 16px; line-height: 1.5;`
           - Add `<ul>` with 4 features. Style: `clear: both; padding-top: 15px; color: #FFFFFF;`
           - Discount: `<span style="background-color: yellow; color: #000000 !important; font-weight: bold; padding: 2px 5px; border: 1px solid black;">{discount_plan}</span>`
        
        5. **The Button**: 
           `<a href="#" style="display: block; width: 250px; margin: 20px auto 0; background-color: #FFFFFF !important; color: #000000 !important; font-size: 20px; font-weight: bold; padding: 15px; text-align: center; text-decoration: none; border-radius: 5px; border: 2px solid #000;">Shop Now - {discount_plan} Off</a>`

        [FINAL REMINDER]: You MUST start with '### STEP 1: DESIGN REASONING'. Failure to provide Step 1 is a violation of instructions.
        """)
    ])
    
    chain = prompt | ollama_client | StrOutputParser()
    
    # Invoke with the three dynamic variables
    full_output = chain.invoke({
        "bike": bike, 
        "ad_theme": ad_theme, 
        "discount_plan": discount_plan
    })

    # Check whether we've received split output with generated HTML and Explanations
    # print(f'Output From llm: {full_output}')


    if "### STEP 2" in full_output:
        parts = full_output.split("### STEP 2")
        thought_process = parts[0].replace("### STEP 1:", "").strip()
        thought_process = thought_process.replace(".", ".\n")
        
        html_output = parts[1].replace("HTML GENERATION (STRICT REQUIREMENTS)", "").strip()
    else:
        thought_process = "Model proceeded directly to output."
        html_output = full_output

    # Final cleanup to remove markdown backticks if the model ignored the system prompt
    clean_html = html_output.replace("```html", "").replace(": HTML GENERATION", "").replace("```", "").strip()
    
    return "", clean_html, thought_process

def generate_ad_campaign(bike: str, discount_plan: str, ad_theme: str, method: str)-> tuple[str, str, str]:

    # Validate input parameters
    if bike == DEFAULT_BIKE:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Bike Type.</span>", DEFAULT_EMPTY_AD_HTML, DEFAULT_REASONING_CHAIN
    
    if discount_plan == DEFAULT_DISCOUNT:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Discount Plan.</span>", DEFAULT_EMPTY_AD_HTML, DEFAULT_REASONING_CHAIN
    
    if ad_theme == DEFAULT_THEME:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Ad Theme.</span>",  DEFAULT_EMPTY_AD_HTML, DEFAULT_REASONING_CHAIN
    
    if method == DEFAULT_METHOD:
        return "<span style='color: red; font-weight: bold; font-size: 22px;'>⚠️ Error: Please select a valid Prompting Method.</span>",  DEFAULT_EMPTY_AD_HTML, DEFAULT_REASONING_CHAIN
    
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
        DEFAULT_EMPTY_AD_HTML, # return default text/colour/size for ad output
        DEFAULT_REASONING_CHAIN # Clear out reasoning chain
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
    
    logic_display = gr.Textbox(value=DEFAULT_REASONING_CHAIN,
                               label="Reasoning Chain", 
                               interactive=False)

    # Item is selected in DropDown list box; instructions prompt removed from list
    bike_type.select(fn=clean_bike_list, inputs=None, outputs=[bike_type, error_label])
    discount.select(fn=clean_discount_list, inputs=None, outputs=[discount, error_label])
    theme.select(fn=clean_theme_list, inputs=None, outputs=[theme, error_label])
    prompt_type.select(fn=clean_method_list, inputs=None, outputs=[prompt_type, error_label])        

    submit_btn.click(
        fn=generate_ad_campaign,
        inputs=[bike_type, discount, theme, prompt_type],
        outputs=[error_label, ad_output, logic_display]
    )      

    reset_btn.click(
        fn=reset_input_controls,
        inputs=None,
        outputs=[bike_type, discount, theme, prompt_type, error_label, ad_output, logic_display]
    )     


# Launch the Gradio app
if __name__ == '__main__':
    AddGenerator.launch()
