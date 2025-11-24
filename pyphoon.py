# 1. เพิ่ม AutoProcessor และ AutoModelForVision2Seq ใน import
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image # 2. เพิ่ม import PIL (Pillow) สำหรับเปิดรูปภาพ

# ลบ 'pipeline' ที่ไม่ได้ใช้ออก
# pipe = pipeline("image-to-text", model="scb10x/typhoon-ocr1.5-2b")

processor = AutoProcessor.from_pretrained("scb10x/typhoon-ocr1.5-2b")
model = AutoModelForVision2Seq.from_pretrained("scb10x/typhoon-ocr1.5-2b")

# 3. กำหนด path และ "เปิด" รูปภาพด้วย PIL
image_path = "/Users/phonsirithabunsri/Desktop/Procapstone/receipts/16c562e3b025bfeaaee173cbef986a67.png_960x960q80.png"
try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: ไม่พบไฟล์รูปภาพที่: {image_path}")
    exit()

# --- นี่คือ "คำสั่ง" ที่คุณถามถึง ---
# เราจะเปลี่ยน "Describe the image." (บรรยายภาพ)
# เป็นคำสั่งที่เจาะจงสำหรับการดึงข้อมูล (OCR Extraction)
ocr_prompt = (
    "กรุณาดึงข้อมูลสำคัญจากใบเสร็จนี้:\n"
    "1. ชื่อร้านค้า หรือ บริษัท\n"
    "2. ราคารวมสุทธิ (Total Amount) : จำนวนสินค้า :ราคาต่อหน่วย :ยอดรวมของสินค้านั้น\n"
    "3. รายการสินค้าทั้งหมด (Items) พร้อมราคาต่อชิ้น"
)
# ---------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # 4. ส่ง "object รูปภาพ" (ที่เปิดแล้ว) เข้าไปแทน "ชื่อไฟล์"
                "image": image, 
            },
            # 5. เปลี่ยนมาใช้ ocr_prompt ที่เรากำหนดไว้
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

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("--- OCR Result (Data Extraction) ---")
print(output_text)