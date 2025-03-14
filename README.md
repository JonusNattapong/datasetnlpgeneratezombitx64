Below is a markdown file for your repository "datasetnlpgeneratezombitx64" that describes the project and its usage. You can adjust as needed.

```markdown name=README.md
# datasetnlpgeneratezombitx64
generate dataset for nlp
# ตัวสร้างชุดข้อมูล NLP

ที่เก็บนี้มีเครื่องมือสำหรับสร้างชุดข้อมูลสังเคราะห์สำหรับงาน NLP ต่างๆ ตัวสร้างจะสร้างชุดข้อมูลในรูปแบบ Hugging Face Datasets ทำให้พร้อมใช้งานกับไลบรารี 🤗 Transformers

## คุณสมบัติ

สร้างชุดข้อมูลสังเคราะห์สำหรับงาน NLP ต่อไปนี้:

- การจัดประเภทข้อความ
- การจัดประเภทโทเค็น (NER) 
- การตอบคำถาม
- การสรุปความ
- การแปลภาษา
- ความคล้ายคลึงของประโยค
- การเติมคำที่ถูกปิดบัง
- การจัดประเภทแบบ Zero-Shot
- การสร้างข้อความ
- การสร้างข้อความแบบ Text2Text
- การตอบคำถามจากตาราง
- การสกัดคุณลักษณะ

## การติดตั้ง

1. โคลนที่เก็บนี้:
```bash
git clone https://github.com/JonusNattapong/datasetnlpgeneratezombitx64.git
cd datasetnlpgeneratezombitx64
```

2. ติดตั้งแพ็คเกจ Python ที่จำเป็น:
```bash
pip install -r requirements.txt
```

## การใช้งาน

เรียกใช้สคริปต์ `main.py` พร้อมงานและตัวเลือกที่ต้องการ

```bash
# สร้างชุดข้อมูลทั้งหมด (ค่าเริ่มต้น)
python main.py

# สร้างชุดข้อมูลเฉพาะประเภท
python main.py --task text_classification

# ระบุจำนวนตัวอย่าง
python main.py --task question_answering --samples 1000

# ระบุไดเรกทอรีผลลัพธ์
python main.py --output my_datasets
```

## การสร้างชุดข้อมูลภาษาไทย

หากต้องการสร้างชุดข้อมูลภาษาไทย ให้ใช้ตัวเลือก `--task translation` และตรวจสอบให้แน่ใจว่าได้เพิ่มคู่ภาษา `("en", "th")` แล้ว ตัวอย่างเช่น:

```bash
python main.py --task translation
```

นอกจากนี้ คุณสามารถแก้ไขเทมเพลตใน `dataset_generator.py` เพื่อเพิ่มตัวอย่างภาษาไทยในงานอื่นๆ เช่น `text_classification`, `sentence_similarity` และ `fill_mask`

## การใช้งานเป็นไลบรารี

คุณสามารถใช้ตัวสร้างชุดข้อมูลในโค้ดของคุณเองได้:
```python
from dataset_generator import NLPDatasetGenerator

# เริ่มต้นตัวสร้าง
generator = NLPDatasetGenerator(output_dir="my_datasets")

# สร้างชุดข้อมูลเฉพาะ
classification_dataset = generator.generate_text_classification(num_samples=500)
qa_dataset = generator.generate_question_answering(num_samples=300)

# สร้างชุดข้อมูลทั้งหมด
generator.generate_all_datasets(samples_per_task=500)
```

## ตัวอย่างชุดข้อมูล

### การจัดประเภทข้อความ
```python
{
  "text": "นี่คือรีวิวเชิงบวกเกี่ยวกับภาพยนตร์",
  "label": 0
}
```

### การตอบคำถาม
```python
{
  "context": "เมืองหลวงของฝรั่งเศสคือปารีส เป็นที่รู้จักกันดีในเรื่องสถานที่สำคัญเช่นหอไอเฟล",
  "question": "อะไรคือเมืองหลวงของฝรั่งเศส?",
  "answers": {"text": "ปารีส", "answer_start": 23}
}
```

## การปรับแต่ง

คุณสามารถปรับแต่งการสร้างชุดข้อมูลโดยแก้ไขเทมเพลตใน `dataset_generator.py` โค้ดถูกออกแบบมาให้ขยายสำหรับตัวอย่างหรือประเภทงานเพิ่มเติมได้ง่าย
``` 