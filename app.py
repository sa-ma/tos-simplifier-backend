from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import tempfile
import os
from PyPDF2 import PdfReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
MAX_PDF_PAGES = 50  # Limit number of pages to process

# Initialize model lazily to avoid memory issues on startup
model = None
tokenizer = None

def get_model():
    """Lazy load the model only when needed"""
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info("Loading model and tokenizer...")
        try:
            model_id = "sa-ma/tos-simplifier"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    return model, tokenizer

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with page limits"""
    reader = PdfReader(pdf_path)
    
    # Limit the number of pages to process
    total_pages = len(reader.pages)
    if total_pages > MAX_PDF_PAGES:
        logger.warning(f"PDF has {total_pages} pages, limiting to first {MAX_PDF_PAGES}")
        pages_to_process = reader.pages[:MAX_PDF_PAGES]
    else:
        pages_to_process = reader.pages
    
    text = []
    for i, page in enumerate(pages_to_process):
        try:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
            logger.info(f"Processed page {i+1}/{len(pages_to_process)}")
        except Exception as e:
            logger.error(f"Error extracting text from page {i+1}: {e}")
            continue
    
    return "\n".join(filter(None, text))

def clean_tos(text):
    """Clean the extracted text"""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"Toggle navigation.*Terms of Service", "", text, flags=re.S)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def make_prompt(text, level="8th"):
    """Create the prompt for summarization"""
    # Limit text length to prevent memory issues
    max_text_length = 8000  # characters
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."
        logger.info(f"Text truncated to {max_text_length} characters")
    
    base = (
        f"[{level}-grade] Summarize this Terms of Service into exactly these four categories:\n"
        "1. Your Duties\n2. Our Rights\n3. Use Limits\n4. Liability & Data Use\n\n"
        "Give up to 3 bullets per category, each starting with a verb and ≤12 words.\n\n"
        "Source Text:\n" + text
    )
    return base

def summarize_pdf(pdf_path, level="8th"):
    """Full pipeline to summarize PDF"""
    try:
        raw = extract_text_from_pdf(pdf_path)
        if not raw.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        cleaned = clean_tos(raw)
        if not cleaned.strip():
            raise ValueError("No meaningful text found after cleaning")
        
        prompt = make_prompt(cleaned, level)
        
        model, tokenizer = get_model()
        
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
    except Exception as e:
        logger.error(f"Error in summarize_pdf: {e}")
        raise

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
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    tmp_file_path = None
    try:
        logger.info(f"Starting to process PDF: {file.filename}")
        
        # Save uploaded file temporarily with chunked reading to prevent memory issues
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            total_size = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                
                tmp_file.write(chunk)
                logger.info(f"Read {total_size} bytes so far...")
            
            tmp_file_path = tmp_file.name
        
        logger.info(f"PDF saved temporarily: {file.filename}, size: {total_size} bytes")
        
        # Process the PDF
        summary_text = summarize_pdf(tmp_file_path, level)
        structured_summary = parse_summary_to_json(summary_text)
        
        logger.info("PDF processing completed successfully")
        return structured_summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Always clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {cleanup_error}")

@app.get("/")
async def root():
    return {"message": "TOS Simplifier API", "endpoints": ["/simplify"]}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test endpoint to verify file upload works without processing"""
    try:
        logger.info(f"Testing file upload: {file.filename}")
        
        # Just read the file in chunks to test upload
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            logger.info(f"Read {total_size} bytes...")
        
        logger.info(f"Upload test successful: {file.filename}, size: {total_size} bytes")
        return {
            "message": "Upload successful",
            "filename": file.filename,
            "size": total_size
        }
        
    except Exception as e:
        logger.error(f"Upload test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}") 
    