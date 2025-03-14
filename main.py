#!/usr/bin/env python3
"""
CLI ตัวสร้างชุดข้อมูล NLP

สคริปต์นี้ให้อินเทอร์เฟซบรรทัดคำสั่งสำหรับสร้างชุดข้อมูลสังเคราะห์
สำหรับงาน NLP ต่างๆ โดยใช้คลาส NLPDatasetGenerator
"""

import argparse
import os
from dataset_generator import NLPDatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="สร้างชุดข้อมูลสังเคราะห์สำหรับงาน NLP")
    parser.add_argument(
        "--task", 
        type=str, 
        choices=[
            "text_classification", 
            "token_classification", 
            "question_answering", 
            "summarization", 
            "translation", 
            "sentence_similarity",
            "fill_mask", 
            "zero_shot_classification", 
            "text_generation", 
            "text2text_generation", 
            "table_qa", 
            "feature_extraction", 
            "all"
        ],
        default="all",
        help="งาน NLP ที่ต้องการสร้างชุดข้อมูล (ค่าเริ่มต้น: all)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=500,
        help="จำนวนตัวอย่างที่จะสร้างต่องาน (ค่าเริ่มต้น: 500)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_datasets",
        help="ไดเรกทอรีผลลัพธ์สำหรับชุดข้อมูลที่สร้าง (ค่าเริ่มต้น: generated_datasets)"
    )
    
    args = parser.parse_args()
    
    # สร้างตัวสร้างชุดข้อมูล
    generator = NLPDatasetGenerator(output_dir=args.output)
    
    # สร้างชุดข้อมูลที่ระบุ
    if args.task == "all":
        generator.generate_all_datasets(samples_per_task=args.samples)
    elif args.task == "text_classification":
        generator.generate_text_classification(num_samples=args.samples)
    elif args.task == "token_classification":
        generator.generate_token_classification(num_samples=args.samples)
    elif args.task == "question_answering":
        generator.generate_question_answering(num_samples=args.samples)
    elif args.task == "summarization":
        generator.generate_summarization(num_samples=args.samples)
    elif args.task == "translation":
        generator.generate_translation(num_samples=args.samples)
    elif args.task == "sentence_similarity":
        generator.generate_sentence_similarity(num_samples=args.samples)
    elif args.task == "fill_mask":
        generator.generate_fill_mask(num_samples=args.samples)
    elif args.task == "zero_shot_classification":
        generator.generate_zero_shot_classification(num_samples=args.samples)
    elif args.task == "text_generation":
        generator.generate_text_generation(num_samples=args.samples)
    elif args.task == "text2text_generation":
        generator.generate_text2text_generation(num_samples=args.samples)
    elif args.task == "table_qa":
        generator.generate_table_qa(num_samples=args.samples)
    elif args.task == "feature_extraction":
        generator.generate_feature_extraction(num_samples=args.samples)
    
    print(f"สร้างชุดข้อมูลสำหรับ'{args.task}' สำเร็จ!")
    print(f"บันทึกผลลัพธ์ไปที่: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()