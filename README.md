# TOS Simplifier FastAPI App

A FastAPI application that simplifies Terms of Service documents using AI. Supports PDF uploads and provides structured summaries organized into key categories.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the FastAPI app:

```bash
uvicorn main:app --reload
```

## API Endpoints

### 1. Text Simplification (`/simplify`)

Upload a PDF file using form data:

**Using curl:**

```bash
curl -X POST "http://localhost:8000/simplify" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_tos_document.pdf" \
  -F "level=8th"
```

**Using Python requests:**

```python
import requests

url = "http://localhost:8000/simplify"
files = {"file": open("your_tos_document.pdf", "rb")}
data = {"level": "8th"}  # Optional: 8th and 12th.

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Parameters:**

- `file`: PDF file to upload (required)
- `level`: Reading level for simplification (optional, default: "8th")

**Response:**

```json
{
  "Your Duties": [
    "Read and follow all rules",
    "Keep account secure",
    "Pay fees on time"
  ],
  "Our Rights": [
    "Change terms anytime",
    "Monitor your activity",
    "Suspend accounts"
  ],
  "Use Limits": [
    "No illegal activities",
    "No sharing accounts",
    "No reverse engineering"
  ],
  "Liability & Data Use": [
    "We're not liable for damages",
    "We collect your data",
    "We may share with partners"
  ]
}
```

### 2. Root Endpoint (`/`)

Get basic API information:

```bash
curl http://localhost:8000/
```

**Response:**

```json
{
  "message": "TOS Simplifier API",
  "endpoints": ["/simplify"]
}
```

## Features

- **PDF Text Extraction**: Automatically extracts text from uploaded PDF files using PyPDF2
- **Text Cleaning**: Removes HTML tags, navigation elements, and extra whitespace
- **AI-Powered Summarization**: Uses a fine-tuned transformer model ([`sa-ma/tos-simplifier`](https://huggingface.co/sa-ma/tos-simplifier)) for intelligent summarization
- **Structured Output**
- **Reading Level Control**: Adjusts complexity (8th grade and 12th grade)

## Technical Details

- **Model**: Uses [`sa-ma/tos-simplifier`](https://huggingface.co/sa-ma/tos-simplifier) transformer model for summarization
- **Text Processing**: Custom cleaning pipeline to remove HTML and navigation elements
- **Generation Parameters**: 
  - Beam search with 4 beams
  - Length penalty of 1.1
  - Maximum output length of 320 tokens
  - Early stopping enabled
- **Input Processing**: Truncates input to 4096 tokens for model compatibility
