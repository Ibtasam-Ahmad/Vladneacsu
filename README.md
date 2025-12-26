# 🏗️ Engineering Drawing JSON Extractor

## Overview

The **Engineering Drawing JSON Extractor** is a tool designed to automatically extract detailed technical information from PDF engineering drawings. It works for single-part or multi-part drawings, handles different designers’ layouts, and outputs all information in a structured **JSON format**. The tool is tailored for steel drawings and supports Romanian technical language.

This is ideal for engineers, project managers, or technical teams who need to quickly digitize and analyze drawing data without manually reviewing every page.

---

## Key Features

* **PDF to Image Conversion**: Converts each page of a PDF drawing into high-quality images for processing.
* **Automatic OCR (Text Recognition)**: Reads and extracts text and tables from drawings using optical character recognition.
* **Comprehensive Data Extraction**:

  * Part information (material, quantity, length, weight, etc.)
  * Holes (count, diameter, location)
  * Dimensions with context (length, spacing, edge distance)
  * BOM (Bill of Materials) tables
  * Notes, annotations, and flags
* **High Accuracy & Confidence Tracking**: Each extracted value includes a confidence level, and uncertain data is clearly flagged.
* **JSON Output**: Structured, standardized output ready for integration into other systems. No guessing—missing or uncertain data is marked clearly.
* **Multi-Page Support**: Processes multiple PDF files and multiple pages within each file.
* **User-Friendly Interface**: Built with Streamlit, offering a simple web interface for uploading PDFs and downloading JSON results.

---

## How It Works (Simplified)

1. **Upload PDF**: The user uploads one or more PDF engineering drawings.
2. **Image Conversion**: Each page of the PDF is converted into an image for analysis.
3. **Text & Table Extraction**: The system extracts text, numbers, and tables using OCR technology.
4. **AI Processing**: A language model analyzes the images and text to extract all relevant information into a structured JSON format.
5. **Results**: The user can view extracted data directly in the app and download a JSON file for further use.

---

## Who Can Use It

* Engineers and designers working with steel parts
* Project managers needing structured drawing data
* Technical teams aiming to automate manual data entry
* Anyone needing consistent, machine-readable information from engineering drawings

---

## Benefits

* **Saves Time**: Reduces hours of manual review and data entry.
* **Minimizes Errors**: Captures all relevant details with confidence levels.
* **Standardized Output**: JSON files can be easily imported into other systems or databases.
* **Supports Romanian Drawings**: Handles local technical language and standards.

---

## Requirements

* A valid **Groq API Key** (for AI-powered data extraction)
* PDF files of engineering drawings
* Web browser to access the Streamlit interface

---

## Usage

1. Enter your **Groq API key**.
2. Upload one or more PDF drawings.
3. Click **“Extract Information”**.
4. View the JSON output directly in the app.
5. Download the JSON file for further use.

---

## Notes

* The tool does not guess values. Missing or unclear information is flagged.
* Works best with clear scans of drawings. Very low-quality PDFs may reduce accuracy.
* JSON structure is designed to be extensible, capturing all available information, including extra notes or annotations.

---

## License

This project is for internal or professional use. Please contact the developer for licensing details if used commercially.

---
