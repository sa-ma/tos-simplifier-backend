from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import tempfile
import os
from PyPDF2 import PdfReader
import logging
import asyncio
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TOS Simplifier API",
    description="API for simplifying Terms of Service documents",
    version="1.0.0"
)

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # Reduced to 5MB limit
MAX_PDF_PAGES = 20  # Reduced to 20 pages
MAX_TEXT_LENGTH = 4000  # Reduced text length limit

# Initialize model lazily to avoid memory issues on startup
model = None
tokenizer = None

def log_memory_usage(stage=""):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"{stage} Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
    except:
        pass  # Ignore if psutil not available

def get_model():
    """Lazy load the model only when needed with timeout handling"""
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info("Loading model and tokenizer...")
        start_time = time.time()
        
        try:
            model_id = "sa-ma/tos-simplifier"
            
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f}s")
            
            # Load model with timeout handling
            logger.info("Loading model (this may take 30-60 seconds)...")
            model_start = time.time()
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model_time = time.time() - model_start
            logger.info(f"Model loaded in {model_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"Total model loading time: {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try with lower precision to save memory
            try:
                logger.info("Trying to load model with lower precision...")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype="float16")
                logger.info("Model loaded with lower precision successfully")
            except Exception as e2:
                logger.error(f"Failed to load model with lower precision: {e2}")
                raise RuntimeError(f"Model loading failed: {e}")
    return model, tokenizer

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with page limits"""
    log_memory_usage("Before PDF reading")
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
            if i % 5 == 0:  # Log memory every 5 pages
                log_memory_usage(f"After page {i+1}")
        except Exception as e:
            logger.error(f"Error extracting text from page {i+1}: {e}")
            continue
    
    log_memory_usage("After PDF text extraction")
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
    max_text_length = MAX_TEXT_LENGTH  # characters
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
    """Full pipeline to summarize PDF with memory optimization"""
    try:
        log_memory_usage("Starting PDF processing")
        
        logger.info("Starting PDF text extraction...")
        raw = extract_text_from_pdf(pdf_path)
        if not raw.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        log_memory_usage("After text extraction")
        
        logger.info("Cleaning extracted text...")
        cleaned = clean_tos(raw)
        if not cleaned.strip():
            raise ValueError("No meaningful text found after cleaning")
        
        log_memory_usage("After text cleaning")
        
        logger.info("Creating prompt...")
        prompt = make_prompt(cleaned, level)
        
        log_memory_usage("Before model loading")
        
        logger.info("Loading model (this may take a moment)...")
        model, tokenizer = get_model()
        
        log_memory_usage("After model loading")
        
        logger.info("Running model inference...")
        # Use smaller batch size and more conservative settings
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048  # Reduced from 4096
        )
        
        log_memory_usage("After tokenization")
        
        # Clear any cached memory before generation
        import gc
        gc.collect()
        
        out_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=2,  # Reduced from 4
            length_penalty=1.0,  # Reduced from 1.1
            max_length=256,  # Reduced from 320
            early_stopping=True,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.eos_token_id
        )
        
        log_memory_usage("After model generation")
        
        logger.info("Decoding model output...")
        result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        
        # Clear memory after processing
        del inputs, out_ids
        gc.collect()
        
        log_memory_usage("After cleanup")
        
        logger.info("PDF processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in summarize_pdf: {e}")
        # Clear memory on error
        import gc
        gc.collect()
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

@app.get("/load-model")
async def load_model():
    """Manually trigger model loading to test timing"""
    try:
        start_time = time.time()
        logger.info("Manual model loading triggered")
        
        model, tokenizer = get_model()
        
        load_time = time.time() - start_time
        logger.info(f"Manual model loading completed in {load_time:.2f}s")
        
        return {
            "status": "success",
            "model_loaded": True,
            "load_time_seconds": load_time,
            "message": f"Model loaded successfully in {load_time:.2f} seconds"
        }
    except Exception as e:
        logger.error(f"Manual model loading failed: {e}")
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }

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
    