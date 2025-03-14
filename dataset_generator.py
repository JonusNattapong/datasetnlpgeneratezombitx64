"""
NLP Dataset Generator (Thai Focus)

This script provides utilities to generate synthetic datasets for various NLP tasks with a focus on Thai language examples:
- Text Classification
- Token Classification
- Table Question Answering
- Question Answering
- Summarization
- Translation (including English-Thai)
- Sentence Similarity
- Fill-Mask
- Zero-Shot Classification
- Text Generation
- Text2Text Generation
- Feature Extraction
"""

import os
import json
import random
import asyncio
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
from datasets import Audio, Image
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from ai_service import AIServiceConfig, AIServiceManager

# Load environment variables
load_dotenv()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class ThaiNLPDatasetGenerator:
    def __init__(self, output_dir="generated_datasets_thai"):
        """
        Initialize the dataset generator focused on Thai.
        
        Args:
            output_dir (str): Directory to save generated datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize AI services
        self.ai_config = AIServiceConfig()
        self.ai_manager = AIServiceManager(self.ai_config)
        
    async def generate_with_ai(self, prompt: str, services: List[str] = None) -> Optional[str]:
        """Generate text using AI services with fallback."""
        if services is None:
            services = ["mistral", "deepseek", "huggingface", "ollama"]
        return await self.ai_manager.generate_with_fallback(prompt, services)
        
    def save_dataset(self, dataset, name):
        """Save the dataset to disk and/or push to Hub."""
        # Save locally
        dataset_path = os.path.join(self.output_dir, name)
        os.makedirs(dataset_path, exist_ok=True)
        dataset.save_to_disk(dataset_path)
        
        # Also save as CSV for easy inspection
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            df.to_csv(os.path.join(dataset_path, f"{split}.csv"), index=False)
            
        print(f"Dataset '{name}' saved to {dataset_path}")
        
    async def generate_text_classification(self, num_samples=1000, num_classes=3, class_names=None):
        """
        Generate a dataset for text classification with a focus on Thai language examples.
        
        Args:
            num_samples (int): Number of samples to generate
            num_classes (int): Number of classes
            class_names (list): Optional list of class names
        """
        if class_names is None:
            class_names = [f"คลาส_{i}" for i in range(num_classes)]
        else:
            num_classes = len(class_names)
            
        # Template sentences for each class (emphasizing Thai templates)
        templates = {
            0: [
                "รีวิวที่ดี: {topic} น่าประทับใจมาก",
                "ฉันชอบ {topic} อย่างมาก",
                "ประสบการณ์กับ {topic} ยอดเยี่ยม",
                "นี่คือรีวิวที่ดีเกี่ยวกับ {topic}"
            ],
            1: [
                "รีวิวที่เป็นกลาง: {topic} ก็พอใช้ได้",
                "ความรู้สึกเฉยๆ ต่อ {topic}",
                "ไม่มีอะไรโดดเด่นใน {topic}",
                "นี่คือรีวิวที่เป็นกลางเกี่ยวกับ {topic}"
            ],
            2: [
                "รีวิวไม่ดี: {topic} แย่มาก",
                "ฉันไม่ชอบ {topic} เลย",
                "{topic} นี้ทำให้ผิดหวัง",
                "นี่คือรีวิวที่ไม่ดีเกี่ยวกับ {topic}"
            ]
        }
        
        # Extend templates for additional classes if needed
        for i in range(3, num_classes):
            templates[i] = [
                f"นี่คือข้อความประเภท {class_names[i]} สำหรับ {{" "topic" "}}",
                f"ตัวอย่างของ {class_names[i]} คือ {{" "topic" "}}"
            ]
            
        topics = ["ภาพยนตร์", "หนังสือ", "สินค้า", "ร้านอาหาร", "บริการ", 
                  "ประสบการณ์", "คอนเสิร์ต", "โรงแรม", "แอปพลิเคชัน", "เกม"]
        
        texts = []
        labels = []
        
        for _ in range(num_samples):
            label = random.randint(0, num_classes-1)
            topic = random.choice(topics)
            
            # Try to use AI for more natural text generation
            prompt = f"Generate a {class_names[label]} review in Thai language about {topic}. Keep it natural and concise."
            try:
                text = await self.generate_with_ai(prompt)
                if not text:
                    # Fallback to template-based generation
                    template = random.choice(templates[min(label, len(templates)-1)])
                    text = template.replace("{topic}", topic)
            except:
                # Fallback to template-based generation
                template = random.choice(templates[min(label, len(templates)-1)])
                text = template.replace("{topic}", topic)
                
            texts.append(text)
            labels.append(label)
            
        # Create dataset
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)], 
                "label": labels[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):], 
                "label": labels[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        dataset = dataset.cast_column("label", ClassLabel(names=class_names))
        
        self.save_dataset(dataset, "text_classification_thai")
        return dataset
        
    def generate_token_classification(self, num_samples=500, entity_types=None):
        """
        Generate a dataset for token classification (NER) with Thai examples.
        
        Args:
            num_samples (int): Number of samples to generate
            entity_types (list): Optional list of entity types
        """
        if entity_types is None:
            entity_types = ["PER", "ORG", "LOC", "DATE"]
            
        # Sample entities for each type including Thai-specific examples
        entities = {
            "PER": ["สมชาย ใจดี", "สิรินทร์ รัตนชัย", "ประวิทย์ สุขใจ", "วราภรณ์ มั่งคั่ง"],
            "ORG": ["บริษัทไทย", "มหาวิทยาลัยกรุงเทพ", "โรงพยาบาลตำรวจ", "ธนาคารแห่งประเทศไทย"],
            "LOC": ["กรุงเทพมหานคร", "เชียงใหม่", "ภูเก็ต", "ชลบุรี"],
            "DATE": ["1 มกราคม 2565", "15 กุมภาพันธ์ 2564", "30 มีนาคม 2563", "10 เมษายน 2566"]
        }
        
        # Template sentences
        templates = [
            "{PER} ทำงานให้กับ {ORG} ที่ตั้งอยู่ใน {LOC} ตั้งแต่ {DATE}.",
            "{ORG} ได้เปิดสาขาใหม่ที่ {LOC} เมื่อ {DATE}.",
            "{PER} ไปเยือน {LOC} เมื่อ {DATE}.",
            "จากข้อมูลของ {PER} บอกว่า {ORG} จะขยายกิจการไปยัง {LOC} ภายใน {DATE}.",
            "การประชุมระหว่าง {PER} กับตัวแทนของ {ORG} จัดขึ้นที่ {LOC} เมื่อ {DATE}."
        ]
        
        texts = []
        token_labels = []
        
        for _ in range(num_samples):
            template = random.choice(templates)
            filled_template = template
            entities_used = {}
            
            for entity_type in entity_types:
                if "{" + entity_type + "}" in filled_template and entity_type in entities:
                    entity_value = random.choice(entities[entity_type])
                    filled_template = filled_template.replace("{" + entity_type + "}", entity_value)
                    entities_used[entity_type] = entity_value
            
            # Create tokens and labels (simple word-level tokenization)
            tokens = []
            labels = []
            for word in filled_template.split():
                clean_word = word.strip(".,;:!?")
                tokens.append(clean_word)
                found_entity = False
                for entity_type, entity_value in entities_used.items():
                    if clean_word in entity_value.split():
                        if clean_word == entity_value.split()[0]:
                            labels.append(f"B-{entity_type}")
                        else:
                            labels.append(f"I-{entity_type}")
                        found_entity = True
                        break
                if not found_entity:
                    labels.append("O")
                if word[-1] in ".,;:!?":
                    tokens.append(word[-1])
                    labels.append("O")
                    
            texts.append(filled_template)
            token_labels.append({"tokens": tokens, "ner_tags": labels})
        
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)],
                "tokens": [t["tokens"] for t in token_labels[:int(num_samples*0.8)]],
                "ner_tags": [t["ner_tags"] for t in token_labels[:int(num_samples*0.8)]]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):],
                "tokens": [t["tokens"] for t in token_labels[int(num_samples*0.8):]],
                "ner_tags": [t["ner_tags"] for t in token_labels[int(num_samples*0.8):]]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "token_classification_thai")
        return dataset
        
    def generate_question_answering(self, num_samples=500):
        """Generate a dataset for extractive question answering with Thai contexts."""
        
        # Thai context templates
        context_templates = [
            "เมืองหลวงของประเทศไทยคือกรุงเทพมหานคร ซึ่งเป็นที่ตั้งของสถานที่สำคัญหลายแห่ง เช่น พระบรมมหาราชวังและวัดพระแก้ว ประเทศไทยตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้.",
            "นักวิทยาศาสตร์ไทยได้ค้นพบวิธีการใหม่ในการรักษาโรคที่หาได้ยากในปัจจุบัน ซึ่งเป็นที่ยอมรับในวงการแพทย์ทั่วโลก.",
            "ประเทศไทยมีความหลากหลายทางวัฒนธรรมและประเพณีที่สืบทอดกันมาเป็นเวลาหลายร้อยปี.",
            "การท่องเที่ยวในไทยมีทั้งชายหาดที่สวยงามและธรรมชาติที่ยอดเยี่ยม พร้อมทั้งอาหารไทยที่ได้รับความนิยมทั่วโลก."
        ]
        
        # Generate QA pairs
        contexts = []
        questions = []
        answers = []
        
        for _ in range(num_samples):
            context = random.choice(context_templates)
            contexts.append(context)
            
            sentences = context.split(" ")
            target_sentence = random.choice(sentences)
            words = target_sentence.split()
            if len(words) < 3:
                continue
            
            question_type = random.choice(["what", "where", "who", "when"])
            
            if question_type == "what":
                answer_text = words[-1]
                question_text = target_sentence.replace(answer_text, "อะไร")
                question_text = f"อะไร {question_text}?"
            elif question_type == "where":
                answer_text = "กรุงเทพมหานคร"
                question_text = "เมืองหลวงของประเทศไทยคือที่ไหน?"
            elif question_type == "who":
                answer_text = words[0]
                question_text = f"{words[0]} คือใคร?"
            else: # when
                answer_text = "เมื่อไม่นานมานี้"
                question_text = "เหตุการณ์ดังกล่าวเกิดขึ้นเมื่อไร?"
            
            answer_start = context.find(answer_text)
            if answer_start == -1:
                answer_start = 0
            
            questions.append(question_text)
            answers.append({"text": answer_text, "answer_start": answer_start})
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "context": contexts[:int(num_samples*0.8)],
                "question": questions[:int(num_samples*0.8)],
                "answers": answers[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "context": contexts[int(num_samples*0.8):],
                "question": questions[int(num_samples*0.8):],
                "answers": answers[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "question_answering_thai")
        return dataset
        
    def generate_summarization(self, num_samples=500):
        """Generate a dataset for text summarization with Thai articles."""
        
        # Thai article templates with summaries
        article_templates = [
            {
                "article": """สภาเมืองกรุงเทพมหานครได้ประชุมเมื่อวันอังคารที่ผ่านมา เพื่อหารือในประเด็นการขยายบริการขนส่งสาธารณะ 
                โดยมีการเสนอขยายเส้นทางรถไฟฟ้าและเพิ่มความถี่ในการบริการรถเมล์ 
                มีข้อกังวลเกี่ยวกับงบประมาณและผลกระทบต่อชุมชนท้องถิ่น 
                ชาวกรุงเทพฯ ส่วนใหญ่ยอมรับแนวคิดนี้โดยเห็นว่าช่วยลดปัญหาจราจร""",
                "summary": "สภาเมืองกรุงเทพฯ พิจารณาขยายบริการขนส่งสาธารณะ พร้อมข้อกังวลด้านงบประมาณและผลกระทบต่อชุมชน"
            },
            {
                "article": """นักวิจัยจากมหาวิทยาลัยไทยได้พัฒนาวัคซีนใหม่ที่มีประสิทธิภาพในการป้องกันโรคที่พบในภูมิภาคเอเชีย 
                ผลการทดลองในกลุ่มตัวอย่างเผยว่าวัคซีนมีอัตราความสำเร็จสูงถึง 95% 
                ข้อข้างเคียงน้อยและปลอดภัยสำหรับผู้ใช้ทุกเพศทุกวัย 
                ทางทีมวิจัยคาดหวังว่าจะได้รับการอนุมัติจากสำนักงานคณะกรรมการอาหารและยาในไม่ช้า""",
                "summary": "นักวิจัยไทยพัฒนาวัคซีนใหม่ที่มีอัตราความสำเร็จ 95% และคาดอนุมัติในเร็ววัน"
            }
        ]
        
        articles = []
        summaries = []
        
        for _ in range(num_samples):
            template = random.choice(article_templates)
            article = template["article"]
            summary = template["summary"]
            articles.append(article)
            summaries.append(summary)
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "article": articles[:int(num_samples*0.8)],
                "summary": summaries[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "article": articles[int(num_samples*0.8):],
                "summary": summaries[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "summarization_thai")
        return dataset
        
    def generate_translation(self, num_samples=500, language_pairs=None):
        """Generate a dataset for machine translation, including English-Thai pairs."""
        if language_pairs is None:
            language_pairs = [("en", "th"), ("th", "en")]
            
        translation_templates = {
            ("en", "th"): [
                {"en": "Hello, how are you?", "th": "สวัสดี คุณเป็นอย่างไรบ้าง?"},
                {"en": "I would like to order a coffee.", "th": "ฉันต้องการสั่งกาแฟ"},
                {"en": "Where is the nearest hospital?", "th": "โรงพยาบาลที่ใกล้ที่สุดอยู่ที่ไหน?"},
                {"en": "What time is it?", "th": "ตอนนี้เวลาเท่าไร?"},
                {"en": "Thank you for your help.", "th": "ขอบคุณสำหรับความช่วยเหลือ"}
            ],
            ("th", "en"): [
                {"th": "สวัสดี คุณเป็นอย่างไรบ้าง?", "en": "Hello, how are you?"},
                {"th": "ฉันต้องการสั่งกาแฟ", "en": "I would like to order a coffee."},
                {"th": "โรงพยาบาลที่ใกล้ที่สุดอยู่ที่ไหน?", "en": "Where is the nearest hospital?"},
                {"th": "ตอนนี้เวลาเท่าไร?", "en": "What time is it?"},
                {"th": "ขอบคุณสำหรับความช่วยเหลือ", "en": "Thank you for your help."}
            ]
        }
        
        all_translations = []
        
        for _ in range(num_samples):
            language_pair = random.choice(language_pairs)
            if language_pair not in translation_templates:
                continue
                
            template = random.choice(translation_templates[language_pair])
            source_lang, target_lang = language_pair
            
            translation = {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source": template[source_lang],
                "target": template[target_lang]
            }
            all_translations.append(translation)
            
        for source_lang, target_lang in language_pairs:
            pair_translations = [t for t in all_translations if t["source_lang"] == source_lang and t["target_lang"] == target_lang]
            if not pair_translations:
                continue
                
            dataset_dict = {
                "train": Dataset.from_dict({
                    "translation": [
                        {source_lang: t["source"], target_lang: t["target"]} 
                        for t in pair_translations[:int(len(pair_translations)*0.8)]
                    ]
                }),
                "test": Dataset.from_dict({
                    "translation": [
                        {source_lang: t["source"], target_lang: t["target"]} 
                        for t in pair_translations[int(len(pair_translations)*0.8):]
                    ]
                })
            }
            
            dataset = DatasetDict(dataset_dict)
            self.save_dataset(dataset, f"translation_{source_lang}_to_{target_lang}_thai")
            
        return dataset
        
    def generate_sentence_similarity(self, num_samples=500):
        """Generate a dataset for sentence similarity with Thai sentence pairs included."""
        
        sentence_pairs = [
            {
                "sentence1": "แมวกำลังเล่นกับเส้นด้าย",
                "sentence2": "แมวกำลังเล่นกับไหมพรม",
                "similarity": 0.9
            },
            {
                "sentence1": "แมวกำลังเล่นกับเส้นด้าย",
                "sentence2": "สุนัขกำลังนอนบนโซฟา",
                "similarity": 0.1
            },
            {
                "sentence1": "The weather is pleasant today.",
                "sentence2": "It is a sunny day.",
                "similarity": 0.85
            },
            {
                "sentence1": "I am going to work.",
                "sentence2": "I am leaving for my office.",
                "similarity": 0.8
            }
        ]
        
        sentence1_list = []
        sentence2_list = []
        similarity_scores = []
        
        for _ in range(num_samples):
            pair = random.choice(sentence_pairs)
            if random.random() > 0.5:
                sentence1 = pair["sentence1"]
                sentence2 = pair["sentence2"]
                similarity = pair["similarity"]
            else:
                sentence1 = random.choice([pair["sentence1"] for pair in sentence_pairs])
                sentence2 = random.choice([pair["sentence2"] for pair in sentence_pairs])
                similarity = random.uniform(0.0, 0.5)
            
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
            similarity_scores.append(similarity)
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "sentence1": sentence1_list[:int(num_samples*0.8)],
                "sentence2": sentence2_list[:int(num_samples*0.8)],
                "similarity": similarity_scores[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "sentence1": sentence1_list[int(num_samples*0.8):],
                "sentence2": sentence2_list[int(num_samples*0.8):],
                "similarity": similarity_scores[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "sentence_similarity_thai")
        return dataset
        
    def generate_fill_mask(self, num_samples=500):
        """Generate a dataset for masked language modeling (fill-mask task) with Thai templates."""
        
        mask_token = "[MASK]"
        templates = [
            f"เมืองหลวงของประเทศไทยคือ {mask_token}.",
            f"อาหารไทยที่อร่อยที่สุดคือ {mask_token}.",
            f"ราชินีแห่งไทยคือ {mask_token}.",
            f"วัฒนธรรมไทยมีเอกลักษณ์ที่ {mask_token}.",
            f"ประเทศไทยมีประวัติศาสตร์ยาวนานถึง {mask_token} ปี."
        ]
        
        answers = [
            ["กรุงเทพมหานคร"],
            ["ต้มยำกุ้ง", "ผัดไทย"],
            ["สมเด็จพระนางเจ้าสิริกิติ์"],
            ["โดดเด่น"],
            ["ราชวงศ์เก่าแก่", "นาน"]
        ]
        
        masked_texts = []
        original_texts = []
        mask_positions = []
        
        for _ in range(num_samples):
            template_idx = random.randint(0, len(templates) - 1)
            template = templates[template_idx]
            answer = random.choice(answers[template_idx])
            mask_pos = template.find(mask_token)
            original_text = template.replace(mask_token, answer)
            masked_texts.append(template)
            original_texts.append(original_text)
            mask_positions.append(mask_pos)
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "masked_text": masked_texts[:int(num_samples*0.8)],
                "original_text": original_texts[:int(num_samples*0.8)],
                "mask_position": mask_positions[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "masked_text": masked_texts[int(num_samples*0.8):],
                "original_text": original_texts[int(num_samples*0.8):],
                "mask_position": mask_positions[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "fill_mask_thai")
        return dataset
        
    def generate_zero_shot_classification(self, num_samples=500):
        """Generate a dataset for zero-shot classification with Thai texts."""
        
        candidate_labels = {
            "ความรู้สึก": ["ดี", "แย่", "เฉยๆ"],
            "หัวข้อ": ["การเมือง", "กีฬา", "เทคโนโลยี", "บันเทิง", "ธุรกิจ"],
            "อารมณ์": ["สุข", "เศร้า", "โกรธ", "กลัว", "ประหลาดใจ"],
            "เจตนา": ["สอบถาม", "ติชม", "แนะนำ", "ชมเชย", "ร้องขอ"]
        }
        
        text_templates = {
            "ความรู้สึก": [
                "ผมประทับใจในบริการของร้านนี้มากๆ.",
                "ประสบการณ์ที่ได้รับนั้นแย่มาก.",
                "อาหารที่สั่งมาอร่อยแต่ส่วนลดไม่คุ้มค่า."
            ],
            "หัวข้อ": [
                "ผลการเลือกตั้งล่าสุดแสดงถึงการเปลี่ยนแปลงในทัศนคติของประชาชน.",
                "การแข่งขันฟุตบอลล่าสุดเต็มไปด้วยความเร้าใจ.",
                "นวัตกรรมใหม่ในเทคโนโลยีเปลี่ยนแปลงโลก."
            ],
            "อารมณ์": [
                "วันนี้ผมรู้สึกมีความสุขมาก!",
                "หลังจากฟังข่าวร้ายใจ ผมรู้สึกเศร้า.",
                "การตอบสนองที่ได้ทำให้รู้สึกโกรธเกินจริง."
            ],
            "เจตนา": [
                "ช่วยบอกวิธีรีเซ็ตรหัสผ่านให้หน่อยได้ไหม?",
                "บริการลูกค้าของที่นี่น่าจะปรับปรุงให้ดีขึ้น.",
                "ชอบการออกแบบของแอปพลิเคชันนี้มาก!"
            ]
        }
        
        texts = []
        label_candidates = []
        categories = []
        
        for _ in range(num_samples):
            category = random.choice(list(candidate_labels.keys()))
            text = random.choice(text_templates[category])
            candidates = candidate_labels[category]
            texts.append(text)
            label_candidates.append(candidates)
            categories.append(category)
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)],
                "candidate_labels": label_candidates[:int(num_samples*0.8)],
                "category": categories[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):],
                "candidate_labels": label_candidates[int(num_samples*0.8):],
                "category": categories[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "zero_shot_classification_thai")
        return dataset
        
    def generate_text_generation(self, num_samples=300):
        """Generate a dataset for text generation with Thai prompts."""
        
        prompt_templates = [
            "กาลครั้งหนึ่งนานมาแล้ว มี", 
            "ในอนาคต เทคโนโลยีจะ", 
            "ถ้าฉันสามารถเดินทางได้ที่ไหนก็ได้ ฉันจะไป", 
            "ความลับแห่งความสุขคือ", 
            "ความท้าทายที่ยิ่งใหญ่ของมนุษยชาติวันนี้คือ"
        ]
        
        prompts = []
        max_lengths = []
        for _ in range(num_samples):
            prompt = random.choice(prompt_templates)
            if random.random() > 0.7:
                words = prompt.split()
                if len(words) > 3:
                    replacements = {"ยิ่งใหญ่": "สำคัญ", "ความลับ": "เคล็ดลับ"}
                    for old, new in replacements.items():
                        if old in words:
                            words[words.index(old)] = new
                            break
                prompt = " ".join(words)
            max_length = random.randint(50, 200)
            prompts.append(prompt)
            max_lengths.append(max_length)
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "prompt": prompts[:int(num_samples*0.8)],
                "max_length": max_lengths[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "prompt": prompts[int(num_samples*0.8):],
                "max_length": max_lengths[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "text_generation_thai")
        return dataset
        
    def generate_text2text_generation(self, num_samples=500):
        """Generate a dataset for text-to-text generation tasks with Thai adaptations."""
        
        templates = [
            {
                "source": "ผมรู้สึกขอบคุณมากถ้าคุณช่วยส่งข้อมูลมาให้โดยเร็ว",
                "target": "ขอความกรุณาส่งข้อมูลมาให้ฉันโดยเร็ว",
                "task": "formal_to_informal"
            },
            {
                "source": "คุณช่วยรีบแก้ไขปัญหานี้ให้หน่อยได้ไหม?",
                "target": "กรุณาดำเนินการแก้ไขปัญหานี้โดยเร็ว",
                "task": "informal_to_formal"
            },
            {
                "source": "นักเรียนทำการบ้านทุกวัน",
                "target": "การบ้านถูกทำโดยนักเรียนทุกวัน",
                "task": "active_to_passive"
            },
            {
                "source": "การบ้านถูกทำโดยนักเรียนทุกวัน",
                "target": "นักเรียนทำการบ้านทุกวัน",
                "task": "passive_to_active"
            },
            {
                "source": "นักวิทยาศาสตร์ค้นพบวิธีแก้ไขปัญหาด้วยวิธีที่ซับซ้อน",
                "target": "นักวิทยาศาสตร์พบวิธีแก้ปัญหาง่ายๆ",
                "task": "simplification"
            }
        ]
        
        sources = []
        targets = []
        tasks = []
        
        for _ in range(num_samples):
            template = random.choice(templates)
            sources.append(template["source"])
            targets.append(template["target"])
            tasks.append(template["task"])
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "source": sources[:int(num_samples*0.8)],
                "target": targets[:int(num_samples*0.8)],
                "task": tasks[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "source": sources[int(num_samples*0.8):],
                "target": targets[int(num_samples*0.8):],
                "task": tasks[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "text2text_generation_thai")
        return dataset
        
    def generate_feature_extraction(self, num_samples=300):
        """Generate a dataset for feature extraction with Thai texts."""
        
        categories = {
            "วิทยาศาสตร์": [
                "ทฤษฎีสัมพัทธภาพอธิบายความสัมพันธ์ระหว่างอวกาศและเวลา.",
                "กลศาสตร์ควอนตัมศึกษาพฤติกรรมของสสารในระดับอะตอม."
            ],
            "ศิลปะ": [
                "ผลงานศิลปะไทยสะท้อนวัฒนธรรมและประวัติศาสตร์อันยาวนาน.",
                "การแสดงพื้นบ้านไทยเต็มไปด้วยสีสันและจังหวะที่ลงตัว."
            ],
            "เทคโนโลยี": [
                "ปัญญาประดิษฐ์มีบทบาทสำคัญในการพัฒนาซอฟต์แวร์.",
                "เทคโนโลยีบล็อกเชนช่วยให้การทำธุรกรรมปลอดภัยยิ่งขึ้น."
            ],
            "ประวัติศาสตร์": [
                "ไทยมีประวัติศาสตร์ที่ยาวนานมากกว่า 700 ปี.",
                "สถาปัตยกรรมของวัดโบราณสะท้อนศิลปะไทยในอดีต."
            ]
        }
        
        texts = []
        text_categories = []
        for _ in range(num_samples):
            category = random.choice(list(categories.keys()))
            text = random.choice(categories[category])
            texts.append(text)
            text_categories.append(category)
            
        dataset_dict = {
            "train": Dataset.from_dict({
                "text": texts[:int(num_samples*0.8)],
                "category": text_categories[:int(num_samples*0.8)]
            }),
            "test": Dataset.from_dict({
                "text": texts[int(num_samples*0.8):],
                "category": text_categories[int(num_samples*0.8):]
            })
        }
        
        dataset = DatasetDict(dataset_dict)
        self.save_dataset(dataset, "feature_extraction_thai")
        return dataset
        
    def generate_all_datasets(self, samples_per_task=500):
        """สร้างชุดข้อมูล NLP ทุกประเภทที่มุ่งเน้นภาษาไทย"""
        print(f"กำลังสร้างชุดข้อมูล NLP ภาษาไทยด้วยตัวอย่าง {samples_per_task} ตัวอย่างต่อแต่ละงาน...")
        
        self.generate_text_classification(num_samples=samples_per_task)
        self.generate_token_classification(num_samples=samples_per_task)
        self.generate_question_answering(num_samples=samples_per_task)
        self.generate_summarization(num_samples=samples_per_task)
        self.generate_translation(num_samples=samples_per_task)
        self.generate_sentence_similarity(num_samples=samples_per_task)
        self.generate_fill_mask(num_samples=samples_per_task)
        self.generate_zero_shot_classification(num_samples=samples_per_task)
        self.generate_text_generation(num_samples=samples_per_task)
        self.generate_text2text_generation(num_samples=samples_per_task)
        self.generate_feature_extraction(num_samples=samples_per_task)
        
        print("สร้างชุดข้อมูลทั้งหมดสำเร็จ!")

async def generate_datasets():
    """Helper function to run async dataset generation"""
    generator = ThaiNLPDatasetGenerator()
    await generator.generate_all_datasets(samples_per_task=100)

# Example usage:
if __name__ == "__main__":
    asyncio.run(generate_datasets())
