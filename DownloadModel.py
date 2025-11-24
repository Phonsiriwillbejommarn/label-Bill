from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image # 2. เพิ่ม import PIL (Pillow) สำหรับเปิดรูปภาพ

# ลบ 'pipeline' ที่ไม่ได้ใช้ออก
pipe = pipeline("image-to-text", model="scb10x/typhoon-in-a-box-v1")

processor = AutoProcessor.from_pretrained("scb10x/typhoon-in-a-box-v1")
model = AutoModelForVision2Seq.from_pretrained("scb10x/typhoon-in-a-box-v1")
