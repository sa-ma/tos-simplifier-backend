from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import tempfile
import os
from PyPDF2 import PdfReader

app = FastAPI()

model_id = "sa-ma/tos-simplifier"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(filter(None, text))

def clean_tos(text):
    """Clean the extracted text"""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"Toggle navigation.*Terms of Service", "", text, flags=re.S)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def make_prompt(text, level="8th"):
    """Create the prompt for summarization"""
    base = (
        f"[{level}-grade] Summarize this Terms of Service into exactly these four categories:\n"
        "1. Your Duties\n2. Our Rights\n3. Use Limits\n4. Liability & Data Use\n\n"
        "Give up to 3 bullets per category, each starting with a verb and ≤12 words.\n\n"
        "Source Text:\n" + text
    )
    return base

def summarize_pdf(pdf_path, level="8th"):
    """Full pipeline to summarize PDF"""
    raw = extract_text_from_pdf(pdf_path)
    cleaned = clean_tos(raw)
    prompt = make_prompt(cleaned, level)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    )
    out_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        length_penalty=1.1,
        max_length=320,
        early_stopping=True
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def parse_summary_to_json(summary_text):
    """Parse the model's summary text into structured JSON"""
    result = {}
    
    lines = summary_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        if ':' in line:
            # Split by the first colon
            parts = line.split(':', 1)
            if len(parts) == 2:
                category_name = parts[0].strip()
                content = parts[1].strip()
                
                if category_name not in result:
                    result[category_name] = []
                
                # Split content by periods to get individual sentences
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                result[category_name].extend(sentences)
    
    return result


@app.post("/simplify")
async def simplify(
    file: UploadFile = File(...),
    level: str = "8th"
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        summary_text = summarize_pdf(tmp_file_path, level)

        structured_summary = parse_summary_to_json(summary_text)

        os.unlink(tmp_file_path)

        return structured_summary

    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/")
async def root():
    return {"message": "TOS Simplifier API", "endpoints": ["/simplify"]} 
    