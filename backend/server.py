import os
# Fix for macOS mutex lock error
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import io
import base64
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Typhoon OCR API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
processor = None
MODEL_ID = "scb10x/typhoon-ocr1.5-2b"

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image string

@app.on_event("startup")
async def startup_event():
    global model, processor
    logger.info(f"Loading model: {MODEL_ID}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.post("/api/process-typhoon")
async def process_image(request: ImageRequest):
    global model, processor
    
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode Base64 image
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            logger.error(f"Invalid image data: {e}")
            raise HTTPException(status_code=400, detail="Invalid image data")

        # OCR Prompt
        ocr_prompt = (
            "กรุณาดึงข้อมูลสำคัญจากใบเสร็จนี้:\n"
            "1. ชื่อร้านค้า หรือ บริษัท (merchant_name)\n"
            "2. เลขที่ใบเสร็จ (receipt_id)\n"
            "3. เลขผู้เสียภาษี (tax_id)\n"
            "4. วันที่ (date) format YYYY-MM-DD\n"
            "5. รายการสินค้าทั้งหมด (items) [name, quantity, price]\n"
            "6. ยอดรวม (total), ภาษี (tax), ส่วนลด (discount)\n"
            "ตอบเป็น JSON format เท่านั้น"
        )
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": ocr_prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        logger.info(f"Raw Output: {output_text}")

        # Attempt to parse JSON from the output
        try:
            # Find JSON-like structure
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = output_text[json_start:json_end]
                data = json.loads(json_str)
                return {"status": "success", "data": data}
            else:
                # Fallback if no JSON found, return raw text in a structure
                return {
                    "status": "success", 
                    "data": {
                        "raw_text": output_text,
                        "merchant_name": None,
                        "items": []
                    }
                }
        except json.JSONDecodeError:
             return {
                    "status": "success", 
                    "data": {
                        "raw_text": output_text,
                        "error": "Failed to parse JSON"
                    }
                }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
