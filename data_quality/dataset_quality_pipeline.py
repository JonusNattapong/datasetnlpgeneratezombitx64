"""
Dataset Quality Pipeline
----------------------
ระบบประมวลผลคุณภาพชุดข้อมูลแบบรวม สำหรับการตรวจสอบและปรับปรุงคุณภาพ
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from .dual_adversarial_validator import DualAdversarialValidator
from .thai_specific_metrics import ThaiSpecificMetrics
from .quality_metrics import QualityMetrics
from .semantic_tree import SemanticTreeExpander
from .data_validator import DatasetValidator

@dataclass
class QualityConfig:
    """การกำหนดค่าสำหรับการตรวจสอบคุณภาพ"""
    min_quality_score: float = 0.8
    max_error_rate: float = 0.1
    semantic_expansion_depth: int = 3
    enable_dual_validation: bool = True
    enable_thai_specific: bool = True
    enable_semantic_tree: bool = True

class DatasetQualityPipeline:
    """ระบบประมวลผลคุณภาพชุดข้อมูลแบบรวม"""
    
    def __init__(
        self,
        ai_service_manager,
        config: Optional[QualityConfig] = None,
        lang: str = "thai"
    ):
        """
        เริ่มต้นระบบประมวลผลคุณภาพ
        
        Args:
            ai_service_manager: ตัวจัดการ AI service
            config: การกำหนดค่าคุณภาพ
            lang: ภาษาของชุดข้อมูล
        """
        self.ai_service = ai_service_manager
        self.config = config or QualityConfig()
        self.lang = lang
        
        # เริ่มต้นระบบย่อย
        self.validator = DatasetValidator()
        self.quality_metrics = QualityMetrics()
        self.thai_metrics = ThaiSpecificMetrics() if lang == "thai" else None
        self.dual_validator = (
            DualAdversarialValidator(ai_service_manager)
            if self.config.enable_dual_validation else None
        )
        self.semantic_expander = (
            SemanticTreeExpander(ai_service_manager)
            if self.config.enable_semantic_tree else None
        )
    
    async def process_dataset(
        self, 
        samples: List[Dict[str, Any]], 
        task_type: str
    ) -> Dict[str, Any]:
        """
        ประมวลผลคุณภาพชุดข้อมูลทั้งหมด
        
        Args:
            samples: รายการตัวอย่างข้อมูล
            task_type: ประเภทงาน
            
        Returns:
            Dict[str, Any]: ผลการประมวลผลคุณภาพ
        """
        results = {
            "validation": await self._validate_samples(samples, task_type),
            "metrics": self._calculate_metrics(samples, task_type)
        }
        
        if self.config.enable_thai_specific and self.thai_metrics:
            results["thai_analysis"] = self._analyze_thai_specifics(samples)
            
        if self.config.enable_semantic_tree and self.semantic_expander:
            results["semantic_expansion"] = await self._expand_semantics(samples)
            
        # สรุปผลและให้คำแนะนำ
        results["summary"] = self._generate_summary(results)
        results["recommendations"] = self._generate_recommendations(results)
        
        return results
    
    async def _validate_samples(
        self, 
        samples: List[Dict[str, Any]], 
        task_type: str
    ) -> Dict[str, Any]:
        """ตรวจสอบความถูกต้องของตัวอย่าง"""
        results = {
            "basic_validation": self.validator.validate_dataset(samples, task_type)
        }
        
        if self.config.enable_dual_validation and self.dual_validator:
            results["dual_validation"] = await self.dual_validator.validate_samples(
                samples, task_type
            )
            
        return results
    
    def _calculate_metrics(
        self, 
        samples: List[Dict[str, Any]], 
        task_type: str
    ) -> Dict[str, Any]:
        """คำนวณเมตริกคุณภาพ"""
        metrics = {}
        
        # เมตริกทั่วไป
        texts = []
        labels = []
        
        for sample in samples:
            if "text" in sample:
                texts.append(sample["text"])
            if "label" in sample:
                labels.append(sample["label"])
        
        if texts:
            metrics["text_diversity"] = self.quality_metrics.calculate_text_diversity(texts)
            
        if labels:
            metrics["class_balance"] = self.quality_metrics.calculate_class_balance(labels)
            
        if task_type == "ner" and "entities" in samples[0]:
            coverage = []
            for sample in samples:
                text_length = len(sample["text"])
                coverage.append(
                    self.quality_metrics.calculate_entity_coverage(
                        sample["entities"], text_length
                    )
                )
            metrics["entity_coverage"] = {
                "mean": sum(coverage) / len(coverage) if coverage else 0,
                "min": min(coverage) if coverage else 0,
                "max": max(coverage) if coverage else 0
            }
            
        if task_type == "summarization":
            metrics["summarization_quality"] = (
                self.quality_metrics.calculate_summarization_quality(
                    [s["article"] for s in samples],
                    [s["summary"] for s in samples]
                )
            )
            
        return metrics
    
    def _analyze_thai_specifics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """วิเคราะห์ลักษณะเฉพาะของภาษาไทย"""
        if not self.thai_metrics:
            return {}
            
        texts = [s.get("text", "") for s in samples if "text" in s]
        if not texts:
            return {}
            
        return {
            "tonal_metrics": [
                self.thai_metrics.calculate_tonal_metrics(text)
                for text in texts
            ],
            "orthography": [
                self.thai_metrics.validate_thai_orthography(text)
                for text in texts
            ],
            "semantic_clusters": self.thai_metrics.calculate_semantic_clusters(texts),
            "formality": [
                self.thai_metrics.analyze_text_formality(text)
                for text in texts
            ]
        }
    
    async def _expand_semantics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ขยายความหมายของข้อมูลแบบรุกขมรรค"""
        if not self.semantic_expander:
            return {}
            
        # รวบรวมคำหลักจากตัวอย่าง
        key_concepts = set()
        for sample in samples:
            if "text" in sample:
                # ตัวอย่างการแยกคำหลัก (ควรพัฒนาให้ซับซ้อนขึ้น)
                words = sample["text"].split()
                key_concepts.update(words[:3])  # ใช้ 3 คำแรกเป็นตัวอย่าง
        
        # ขยายแนวคิดหลัก
        expansions = {}
        for concept in key_concepts:
            try:
                expansion = await self.semantic_expander.expand_concept(
                    concept, 
                    self.config.semantic_expansion_depth
                )
                expansions[concept] = expansion
            except Exception as e:
                print(f"ข้อผิดพลาดในการขยายแนวคิด '{concept}': {str(e)}")
                
        return {
            "key_concepts": list(key_concepts),
            "expansions": expansions
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างสรุปผลการวิเคราะห์"""
        validation = results.get("validation", {})
        metrics = results.get("metrics", {})
        
        # คำนวณคะแนนคุณภาพรวม
        quality_scores = []
        
        # คะแนนจากการตรวจสอบพื้นฐาน
        if "basic_validation" in validation:
            basic_score = validation["basic_validation"].get("validity_ratio", 0)
            quality_scores.append(basic_score)
            
        # คะแนนจาก dual validation
        if "dual_validation" in validation:
            dual_metrics = validation["dual_validation"].get("overall_metrics", {})
            if "average_score" in dual_metrics:
                quality_scores.append(dual_metrics["average_score"])
                
        # คะแนนจากความหลากหลายของข้อความ
        if "text_diversity" in metrics:
            diversity = metrics["text_diversity"]
            if "uniqueness_ratio" in diversity:
                quality_scores.append(diversity["uniqueness_ratio"])
                
        # คำนวณคะแนนเฉลี่ย
        overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "overall_quality_score": overall_score,
            "meets_quality_threshold": overall_score >= self.config.min_quality_score,
            "error_rate": 1 - overall_score,
            "quality_scores_breakdown": {
                "validation_score": basic_score if "basic_validation" in validation else None,
                "dual_validation_score": dual_metrics["average_score"] if "dual_validation" in validation else None,
                "diversity_score": diversity["uniqueness_ratio"] if "text_diversity" in metrics else None
            }
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """สร้างคำแนะนำในการปรับปรุง"""
        recommendations = []
        summary = results.get("summary", {})
        metrics = results.get("metrics", {})
        
        # ตรวจสอบคะแนนคุณภาพรวม
        if not summary.get("meets_quality_threshold", False):
            recommendations.append(
                "ควรปรับปรุงคุณภาพข้อมูลโดยรวม เนื่องจากคะแนนต่ำกว่าเกณฑ์ที่กำหนด "
                f"({summary.get('overall_quality_score', 0):.2f} < {self.config.min_quality_score})"
            )
        
        # ตรวจสอบความสมดุลของคลาส
        if "class_balance" in metrics:
            balance = metrics["class_balance"]
            if balance.get("gini_impurity", 0) > 0.5:
                recommendations.append(
                    "ควรปรับปรุงความสมดุลของคลาส เนื่องจากมีความไม่สมดุลสูง"
                )
        
        # ตรวจสอบความหลากหลายของข้อความ
        if "text_diversity" in metrics:
            diversity = metrics["text_diversity"]
            if diversity.get("uniqueness_ratio", 1) < 0.8:
                recommendations.append(
                    "ควรเพิ่มความหลากหลายของข้อความ เนื่องจากมีข้อความที่ซ้ำกันมาก"
                )
        
        # คำแนะนำสำหรับภาษาไทย
        if "thai_analysis" in results:
            thai = results["thai_analysis"]
            
            # ตรวจสอบปัญหาการสะกด
            if any(
                analysis.get("misspellings", [])
                for analysis in thai.get("orthography", [])
            ):
                recommendations.append(
                    "พบปัญหาการสะกดภาษาไทย ควรตรวจสอบและแก้ไขการสะกดให้ถูกต้อง"
                )
            
            # ตรวจสอบการใช้วรรณยุกต์
            tonal_issues = False
            for metrics in thai.get("tonal_metrics", []):
                if metrics.get("double_tone_instances", []):
                    tonal_issues = True
                    break
            if tonal_issues:
                recommendations.append(
                    "พบปัญหาการใช้วรรณยุกต์ ควรตรวจสอบการใช้วรรณยุกต์ให้ถูกต้อง"
                )
        
        return recommendations
