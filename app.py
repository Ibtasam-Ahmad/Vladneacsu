from PIL import Image
import streamlit as st
import os
import fitz
import json
import base64
import re
import tempfile
from groq import Groq
import anthropic
from openai import OpenAI
import camelot
from dotenv import load_dotenv
import json5

load_dotenv()

MAX_PIXELS = 33160500

# ========================= MODEL CONFIGURATIONS =========================

MODEL_OPTIONS = {
    'openai': [
        'gpt-5-mini',
        'gpt-5-nano',
        'gpt-4.1-mini',
        'gpt-4.1-nano',
        'gpt-4o-mini',
        'gpt-5',
        'gpt-5.1',
        'gpt-4.1',
        'gpt-4o'
    ],
    'grok': [
        'llama-3.3-70b-versatile',
        'meta-llama/llama-4-maverick-17b-128e-instruct',
        'meta-llama/llama-4-scout-17b-16e-instruct',
        'moonshotai/kimi-k2-instruct-0905',
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b'
    ],
    'claude': [
        'claude-sonnet-4-5-20250929',
        'claude-haiku-4-5-20251001',
        'claude-opus-4-5-20251101',
        'claude-sonnet-4-5'
    ]
}

DEFAULT_MODELS = {
    'openai': 'gpt-4o-mini',
    'grok': 'llama-3.3-70b-versatile',
    'claude': 'claude-sonnet-4-5'
}

# ========================= SETTINGS MANAGEMENT =========================

def init_session_state():
    """Initialize session state for settings"""
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'ai_provider': 'openai',
            'api_key': '',
            'model': DEFAULT_MODELS['openai']
        }
    
    # Load from query params (persists across page reloads)
    if 'settings_loaded' not in st.session_state:
        params = st.query_params
        if 'settings' in params:
            try:
                saved_settings = json.loads(params['settings'])
                st.session_state.settings.update(saved_settings)
            except:
                pass
        st.session_state.settings_loaded = True

def save_settings():
    """Save settings to query params for persistence"""
    # Update session state from widgets FIRST
    update_settings_from_widgets()
    st.query_params['settings'] = json.dumps(st.session_state.settings)

def update_settings_from_widgets():
    """Update settings from widget values"""
    # Get current values from widgets if they exist
    if 'ai_provider_widget' in st.session_state:
        st.session_state.settings['ai_provider'] = st.session_state.ai_provider_widget
    if 'model_widget' in st.session_state:
        st.session_state.settings['model'] = st.session_state.model_widget
    if 'api_key_widget' in st.session_state:
        st.session_state.settings['api_key'] = st.session_state.api_key_widget

def get_client(provider, api_key):
    """Get the appropriate client based on provider selection"""
    if not api_key:
        return None
    
    if provider == 'openai':
        return OpenAI(api_key=api_key)
    elif provider == 'grok':
        return Groq(api_key=api_key)
    elif provider == 'claude':
        return anthropic.Anthropic(api_key=api_key)
    else:
        st.error(f"Unknown provider: {provider}")
        return None

# ========================= CORE LOGIC =========================

def pdf_to_images(pdf_path, dpi=150):
    """
    Convert PDF pages to images and resize if too large.
    """
    doc = fitz.open(pdf_path)
    paths = []

    for i, page in enumerate(doc):
        img = page.get_pixmap(dpi=dpi)
        img_path = f"{pdf_path}_page_{i+1}.jpg"
        img.save(img_path)

        # Resize if too big
        img_pil = Image.open(img_path)
        width, height = img_pil.size
        if width * height > MAX_PIXELS:
            scale = (MAX_PIXELS / (width * height)) ** 0.5
            new_size = (int(width * scale), int(height * scale))
            img_pil = img_pil.resize(new_size, Image.LANCZOS)
            img_pil.save(img_path)

        paths.append(img_path)

    return paths


def extract_ocr(pdf_path, page):
    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page),
            flavor="stream",
            edge_tol=500
        )
        return "\n".join(t.df.to_string(index=False) for t in tables)
    except:
        return ""


def encode_img(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_json(image, ocr, settings):
    """Modified to use settings from session state"""
    api_key = settings['api_key']
    provider = settings['ai_provider']
    model = settings['model']
    
    if not api_key:
        raise ValueError("API key is required. Please enter your API key in the settings.")
    
    client = get_client(provider, api_key)
    if not client:
        raise ValueError(f"Failed to initialize client for {provider}")
    
    prompt = f"""
        You are an ENGINEERING DRAWING INFORMATION EXTRACTION ENGINE.

        TASK:
        Extract ALL available information from the drawing image and the OCR text provided.
        The system must handle single-part or multi-part drawings, different designers, and varying layouts.

        ABSOLUTE RULES:
        - Extract EVERYTHING: all text, numbers, dimensions, tables, annotations.
        - Extract EACH PART separately, even if multiple parts exist.
        - Romanian technical language is expected.
        - Do NOT guess values. If a value cannot be confidently extracted → use null and flag it.
        - Correlate BOM tables with parts using part marks.
        - Identify holes using notation like:
        - "12*D22" → 12 holes Ø22
        - "2*D18" → 2 holes Ø18
        - Identify hole locations if possible (e.g., web / flange / unknown).
        - Include all numeric data with context if derivable (length, area, weight, spacing, etc.).
        - Capture any notes, annotations, flags, or extra information not explicitly listed below.
        - If there is any information not included in the JSON template, create new fields under "extra_info" for that part or drawing.

        RETURN:
        - ONE JSON object only, using this exact structure:

        {{
        "drawing_meta": {{
            "drawing_id": null,
            "scale": null,
            "date": null,
            "project": null,
            "language": "ro",
            "confidence": "high|medium|low",
            "additional_info": {{}}
        }},

        "parts": [
            {{
            "part_mark": null,
            "profile_type": null,
            "material": null,
            "quantity": null,
            "length_mm": null,
            "area": null,
            "weight": null,
            "subassembly": null,

            "holes": [
                {{
                "count": null,
                "diameter_mm": null,
                "location": "web|flange|unknown",
                "source": "annotation|dimension|table",
                "confidence": "high|medium|low",
                "additional_info": {{}}
                }}
            ],

            "dimensions": [
                {{
                "value": null,
                "unit": "mm",
                "context": "length|spacing|edge-distance|unknown",
                "confidence": "high|medium|low",
                "additional_info": {{}}
                }}
            ],

            "notes": [],
            "flags": [],
            "additional_info": {{}}
            }}
        ],

        "bom_tables": [
            {{
            "title": null,
            "rows": [
                {{
                "part_mark": null,
                "quantity": null,
                "subassembly": null,
                "other_columns": {{}},
                "additional_info": {{}}
                }}
            ]
            }}
        ],

        "annotations": [],
        "all_text": [],
        "all_numbers": [],
        "flags": [],
        "additional_info": {{}}
        }}

        INSTRUCTIONS:
        - Fill as many fields as possible using both OCR text and visual drawing elements.
        - Maintain confidence levels for each extracted value.
        - Flag any missing, uncertain, or conflicting information.
        - Ensure BOM tables are correlated with parts using part marks.
        - Capture any additional information not explicitly listed in the JSON under "additional_info" at the appropriate level (drawing_meta, parts, holes, dimensions, BOM, etc.).
        - Preserve all text and numeric data for potential future reference.

        OCR TEXT INPUT:
        {ocr}
    """

    try:
        if provider == 'openai':
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_img(image)}"}
                        }
                    ]
                }]
            )
            return response.choices[0].message.content
        
        elif provider == 'grok':
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": f"Image Data: data:image/jpeg;base64,{encode_img(image)}"}
                ]
            )
            return response.output_text.strip()
        
        elif provider == 'claude':
            # Claude implementation with correct base64 image format
            message = client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_img(image)
                                }
                            }
                        ]
                    }
                ]
            )
            return message.content[0].text
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower():
            raise ValueError(f"Authentication failed. Please check your {provider.upper()} API key.")
        elif "404" in error_msg:
            raise ValueError(f"Model '{model}' not found. Please select a different model.")
        elif "rate limit" in error_msg.lower():
            raise ValueError(f"Rate limit exceeded. Please try again later.")
        elif "quota" in error_msg.lower():
            raise ValueError(f"Quota limit exceeded. Please recharge your account.")
        else:
            raise ValueError(f"API call failed: {error_msg}")


def sanitize_json(obj):
    """
    Recursively walk JSON-like object and fix invalid numbers.
    - Converts anything that is not int/float/bool/null/list/dict to string
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, (int, float)) or obj is None or isinstance(obj, bool):
        return obj
    else:
        # Anything else (e.g., 1:5, PL6*70) becomes string
        return str(obj)


def parse_json(txt):
    """
    Parse AI-generated JSON robustly.
    - Removes ```json fences
    - Uses json5 for tolerant parsing
    - Falls back to raw text if parsing fails
    """
    # Strip ```json fences
    txt = re.sub(r"^```(?:json)?\s*", "", txt.strip())
    txt = re.sub(r"\s*```$", "", txt.strip())

    # Try extracting first JSON-like block
    match = re.search(r"\{.*\}", txt, re.DOTALL)
    json_str = match.group() if match else txt

    try:
        parsed = json5.loads(json_str)
        return sanitize_json(parsed)
    except Exception as e:
        return {"error": str(e), "raw_output": json_str}


def process_pdf(pdf_path, settings):
    images = pdf_to_images(pdf_path)
    result = []

    for i, img in enumerate(images, 1):
        ocr = extract_ocr(pdf_path, i)
        raw = extract_json(img, ocr, settings)
        parsed = parse_json(raw)
        result.append({
            "page": i,
            "data": parsed
        })

    return result

# ========================= STREAMLIT UI =========================

st.set_page_config(
    page_title="Engineering Drawing JSON Extractor",
    layout="wide"
)

# Initialize session state
init_session_state()

# ========================= SIDEBAR SETTINGS =========================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Configure your AI provider and API key")
    
    # Make sure settings are updated from session state
    update_settings_from_widgets()
    
    # Provider selection - use simple keys with _widget suffix
    provider_options = ['openai', 'grok', 'claude']
    provider_index = provider_options.index(st.session_state.settings['ai_provider']) if st.session_state.settings['ai_provider'] in provider_options else 0
    
    provider = st.selectbox(
        "AI Provider",
        options=provider_options,
        format_func=lambda x: {
            'openai': 'OpenAI',
            'grok': 'Grok (via Groq)',
            'claude': 'Claude (Anthropic)'
        }[x],
        index=provider_index,
        key='ai_provider_widget',
        on_change=save_settings
    )
    
    # Update model when provider changes
    if st.session_state.settings['ai_provider'] != provider:
        st.session_state.settings['model'] = DEFAULT_MODELS[provider]
        st.session_state.settings['ai_provider'] = provider
        save_settings()
    
    # Model selection based on provider
    current_model = st.session_state.settings['model']
    model_options = MODEL_OPTIONS[provider]
    model_index = model_options.index(current_model) if current_model in model_options else 0
    
    model = st.selectbox(
        "Model",
        options=model_options,
        index=model_index,
        key='model_widget',
        on_change=save_settings
    )
    
    # API Key input
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder=f"Enter your {provider.upper()} API key",
        value=st.session_state.settings['api_key'],
        key='api_key_widget',
        on_change=save_settings
    )
    
    # Provider-specific info
    st.divider()
    st.markdown("### ℹ️ Provider Info")
    
    if provider == 'openai':
        st.info("OpenAI Vision models support image analysis. GPT-4o series recommended.")
    elif provider == 'grok':
        st.info("Grok via Groq API. Supports various open-source models with fast inference.")
    elif provider == 'claude':
        st.info("Claude models from Anthropic. Claude Sonnet recommended for balance of cost/performance.")
    
    st.divider()
    
    # Quick instructions
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Select your AI provider
    2. Choose a model
    3. Enter your API key
    4. Upload PDF drawings
    5. Click 'Extract Information'
    
    ⚠️ **Note**: Settings persist in URL until page refresh.
    """)
    
    # Test API connection button
    if st.button("🔗 Test Connection", type="secondary"):
        # Update settings before testing
        update_settings_from_widgets()
        if not st.session_state.settings['api_key']:
            st.error("Please enter an API key first.")
        else:
            try:
                client = get_client(
                    st.session_state.settings['ai_provider'],
                    st.session_state.settings['api_key']
                )
                if client:
                    st.success(f"✅ Connected to {st.session_state.settings['ai_provider'].upper()} successfully!")
                else:
                    st.error("Failed to initialize client.")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
    
    # Clear settings button
    if st.button("🔄 Clear Settings"):
        st.session_state.settings = {
            'ai_provider': 'openai',
            'api_key': '',
            'model': DEFAULT_MODELS['openai']
        }
        st.query_params.clear()
        st.rerun()

# ========================= MAIN CONTENT =========================
st.title("🏗️ Engineering Drawing JSON Extractor")
st.markdown("""
This app extracts **ALL technical information** from steel drawings:
- Single-part & multi-part drawings
- Different designers & layouts
- Romanian language support
- JSON-only output (no guessing, flagged uncertainty)
""")

# Update settings from widgets before displaying
update_settings_from_widgets()

# Display current settings in main area
with st.expander("🔍 Current Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        provider_display = st.session_state.settings['ai_provider'].upper()
        st.metric("Provider", provider_display)
    with col2:
        model_display = st.session_state.settings['model']
        st.metric("Model", model_display)
    with col3:
        api_key_status = "✅ Set" if st.session_state.settings['api_key'] else "❌ Not Set"
        st.metric("API Key", api_key_status)

# FILE UPLOAD
uploaded_files = st.file_uploader(
    "📄 Upload PDF drawing(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🚀 Extract Information", type="primary"):
    # Update settings from widgets before processing
    update_settings_from_widgets()
    settings = st.session_state.settings
    
    if not settings['api_key']:
        st.error("⚠️ API key is required. Please enter your API key in the settings sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        # Process each uploaded PDF
        for uploaded_pdf in uploaded_files:
            st.divider()
            st.subheader(f"📄 Processing: {uploaded_pdf.name}")
            
            # Display processing info
            with st.status(f"Extracting from {uploaded_pdf.name}...", expanded=True) as status:
                st.write(f"**Provider:** {settings['ai_provider'].upper()}")
                st.write(f"**Model:** {settings['model']}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(uploaded_pdf.read())
                    pdf_path = tmp_pdf.name

                try:
                    result = process_pdf(pdf_path, settings)
                    status.update(label="✅ Extraction completed!", state="complete")
                except Exception as e:
                    status.update(label="❌ Extraction failed", state="error")
                    st.error(f"Extraction failed: {e}")
                    continue

            # SHOW RESULTS
            st.success(f"✅ Successfully extracted {len(result)} pages from {uploaded_pdf.name}")

            st.markdown("### 📦 Extracted JSON")
            
            # Display in tabs for better organization
            tab1, tab2 = st.tabs(["📊 Formatted View", "📋 Raw JSON"])
            
            with tab1:
                for page_result in result:
                    with st.expander(f"Page {page_result['page']}", expanded=False):
                        st.json(page_result['data'])
            
            with tab2:
                st.code(json.dumps(result, indent=2), language='json')

            # DOWNLOAD
            json_bytes = json.dumps(result, indent=2).encode("utf-8")
            st.download_button(
                label="⬇️ Download JSON",
                data=json_bytes,
                file_name=f"{uploaded_pdf.name.split('.')[0]}_extracted.json",
                mime="application/json",
                key=f"download_{uploaded_pdf.name}"
            )

            # CLEANUP
            try:
                os.remove(pdf_path)
            except:
                pass

st.divider()
st.caption("⚙️ Vision + OCR + LLM | No hardcoded fields | Engineering-grade extraction")

# Add installation requirements
with st.expander("📦 Installation Requirements", expanded=False):
    st.markdown("""
    ### Required Packages:
    ```bash
    pip install streamlit pillow pymupdf camelot-py openai groq anthropic json5 python-dotenv
    ```
    
    ### For PDF table extraction (optional but recommended):
    ```bash
    # Install GhostScript for Camelot
    # Ubuntu/Debian:
    sudo apt-get install ghostscript
    
    # macOS:
    brew install ghostscript
    
    # Windows:
    # Download from: https://www.ghostscript.com/download/gsdnld.html
    ```
    """)
