"""
NLP Dataset Generator (Thai Focus)
--------------------------------
ระบบสร้างชุดข้อมูล NLP อัตโนมัติเฉพาะภาษาไทยพร้อมการตรวจสอบคุณภาพขั้นสูง
"""

import os
import json
import random
import asyncio
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
from datasets import Audio, Image
from typing import List, Optional, Dict, Any, Union, Tuple
from dotenv import load_dotenv
from ai_service import AIServiceConfig, AIServiceManager
from data_quality.dataset_quality_pipeline import DatasetQualityPipeline, QualityConfig
from data_quality.semantic_tree import SemanticTreeExpander
from data_quality.dual_adversarial_validator import DualAdversarialValidator

# Load environment variables
load_dotenv()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class ThaiNLPDatasetGenerator:
    def __init__(self, output_dir="generated_datasets_thai", quality_config: Optional[QualityConfig] = None):
        """
        Initialize the dataset generator with advanced quality checks.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize core components
        self.ai_config = AIServiceConfig()
        self.ai_manager = AIServiceManager(self.ai_config)
        
        # Initialize quality pipeline
        self.quality_config = quality_config or QualityConfig(
            min_quality_score=0.8,
            semantic_expansion_depth=3,
            enable_dual_validation=True,
            enable_thai_specific=True,
            enable_semantic_tree=True
        )
        
        # Initialize quality components
        self.quality_pipeline = DatasetQualityPipeline(
            self.ai_manager,
            config=self.quality_config,
            lang="thai"
        )
        self.semantic_expander = SemanticTreeExpander(self.ai_manager)
        self.dual_validator = DualAdversarialValidator(self.ai_manager)

    async def save_dataset(self, dataset: DatasetDict, name: str, task_type: str) -> Dict[str, Any]:
        """
        Save dataset with quality analysis and semantic expansion.
        """
        # Create output directory
        dataset_path = os.path.join(self.output_dir, name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Run quality analysis
        quality_results = await self.quality_pipeline.process_dataset(dataset["train"], task_type)
        
        # Save quality report
        with open(os.path.join(dataset_path, "quality_report.json"), "w", encoding="utf-8") as f:
            json.dump(quality_results, f, ensure_ascii=False, indent=2)
            
        # Perform semantic expansion if enabled
        if self.quality_config.enable_semantic_tree and "text" in dataset["train"].features:
            semantic_results = await self._analyze_semantics(dataset["train"]["text"][:10])
            with open(os.path.join(dataset_path, "semantic_analysis.json"), "w", encoding="utf-8") as f:
                json.dump(semantic_results, f, ensure_ascii=False, indent=2)
                
        # Save dataset with quality metrics
        enhanced_dataset = self._add_quality_metrics(dataset, quality_results)
        enhanced_dataset.save_to_disk(dataset_path)
        
        # Save CSV with quality metrics
        for split in enhanced_dataset.keys():
            df = enhanced_dataset[split].to_pandas()
            df.to_csv(os.path.join(dataset_path, f"{split}.csv"), index=False)
            
        print(f"Dataset '{name}' saved with quality analysis to {dataset_path}")
        return quality_results
        
    async def _analyze_semantics(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text semantics using semantic tree expansion."""
        results = {}
        for text in texts:
            try:
                expansion = await self.semantic_expander.expand_concept(text, max_depth=2)
                results[text] = expansion
            except Exception as e:
                results[text] = {"error": str(e)}
        return results
        
    def _add_quality_metrics(self, dataset: DatasetDict, quality_results: Dict[str, Any]) -> DatasetDict:
        """Add quality metrics to dataset."""
        enhanced = DatasetDict()
        
        for split in dataset.keys():
            split_data = dataset[split].to_dict()
            
            if split == "train":
                # Add overall quality metrics
                quality_scores = quality_results["summary"]["quality_scores_breakdown"]
                for metric, score in quality_scores.items():
                    if score is not None:
                        split_data[f"quality_{metric}"] = [score] * len(dataset[split])
                        
                # Add validation feedback if available
                if "validation_results" in quality_results:
                    feedback = quality_results["validation_results"]
                    if isinstance(feedback, list) and len(feedback) == len(dataset[split]):
                        split_data["validation_feedback"] = feedback
                        
            enhanced[split] = Dataset.from_dict(split_data)
            
        return enhanced

    async def generate_all_datasets(
        self,
        samples_per_task: int = 500,
        min_quality_score: float = 0.8,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate all dataset types with quality validation and retries.
        """
        print(f"กำลังสร้างชุดข้อมูล NLP ภาษาไทยด้วยตัวอย่าง {samples_per_task} ตัวอย่างต่องาน...")
        
        generation_tasks = [
            ("text_classification", self.generate_text_classification),
            ("token_classification", self.generate_token_classification),
            ("question_answering", self.generate_question_answering),
            ("summarization", self.generate_summarization),
            ("translation", self.generate_translation),
            ("sentence_similarity", self.generate_sentence_similarity),
            ("fill_mask", self.generate_fill_mask),
            ("zero_shot_classification", self.generate_zero_shot_classification),
            ("text_generation", self.generate_text_generation),
            ("text2text_generation", self.generate_text2text_generation),
            ("feature_extraction", self.generate_feature_extraction)
        ]
        
        results = {}
        for task_name, generator_func in generation_tasks:
            attempts = 0
            while attempts < max_retries:
                try:
                    print(f"\nกำลังสร้างชุดข้อมูล {task_name}...")
                    dataset = await generator_func(num_samples=samples_per_task)
                    quality_results = await self.save_dataset(dataset, f"{task_name}_thai", task_name)
                    
                    if quality_results["summary"]["overall_quality_score"] >= min_quality_score:
                        results[task_name] = {
                            "status": "success",
                            "quality_score": quality_results["summary"]["overall_quality_score"],
                            "recommendations": quality_results["recommendations"]
                        }
                        break
                    else:
                        print(f"คุณภาพชุดข้อมูลต่ำกว่าเกณฑ์ กำลังลองใหม่... (ครั้งที่ {attempts + 1})")
                        attempts += 1
                except Exception as e:
                    print(f"เกิดข้อผิดพลาดในการสร้าง {task_name}: {str(e)}")
                    attempts += 1
            
            if attempts == max_retries:
                results[task_name] = {
                    "status": "failed",
                    "error": "ไม่สามารถสร้างชุดข้อมูลที่มีคุณภาพตามเกณฑ์ได้หลังจากลองหลายครั้ง"
                }
        
        # Save generation report
        report_path = os.path.join(self.output_dir, "generation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results

async def main():
    """Helper function to run dataset generation"""
    generator = ThaiNLPDatasetGenerator()
    results = await generator.generate_all_datasets(samples_per_task=100)
    print("\nผลการสร้างชุดข้อมูล:")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
