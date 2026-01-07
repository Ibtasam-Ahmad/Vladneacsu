from PIL import Image
import streamlit as st
import os
import fitz
import json
import base64
import re
import tempfile
from groq import Groq
import camelot
from dotenv import load_dotenv
import json5  


load_dotenv()
api_key = st.secrets["GROQ_API_KEY"] or os.getenv("GROQ_API_KEY")

MODEL = "llama-3.3-70b-versatile"
MAX_PIXELS = 33166500

# ========================= CORE LOGIC (UNCHANGED) =========================

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


def extract_json(image, ocr, api_key):
    client = Groq(api_key=api_key)

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

    res = client.chat.completions.create(
        model=MODEL,
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
    return res.choices[0].message.content

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


def process_pdf(pdf_path, api_key):
    images = pdf_to_images(pdf_path)
    result = []

    for i, img in enumerate(images, 1):
        ocr = extract_ocr(pdf_path, i)
        raw = extract_json(img, ocr, api_key)
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

st.title("🏗️ Engineering Drawing JSON Extractor")
st.markdown("""
This app extracts **ALL technical information** from steel drawings:
- Single-part & multi-part drawings
- Different designers & layouts
- Romanian language support
- JSON-only output (no guessing, flagged uncertainty)
""")


# FILE UPLOAD
uploaded_files = st.file_uploader(
    "📄 Upload PDF drawing(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🚀 Extract Information"):
    if not api_key:
        st.error("API KEY not found.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        for uploaded_pdf in uploaded_files:
            st.divider()
            st.subheader(f"📄 Processing: {uploaded_pdf.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(uploaded_pdf.read())
                pdf_path = tmp_pdf.name

            with st.spinner("Extracting data..."):
                try:
                    result = process_pdf(pdf_path, api_key)
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    continue

            # SHOW RESULTS
            st.success("Extraction completed.")

            st.markdown("### 📦 Extracted JSON")
            st.json(result)

            # DOWNLOAD
            json_bytes = json.dumps(result, indent=2).encode("utf-8")
            st.download_button(
                label="⬇️ Download JSON",
                data=json_bytes,
                file_name=f"{uploaded_pdf.name}_extracted.json",
                mime="application/json"
            )

            # CLEANUP
            os.remove(pdf_path)

st.divider()
st.caption("⚙️ Vision + OCR + LLM | No hardcoded fields | Engineering-grade extraction")
