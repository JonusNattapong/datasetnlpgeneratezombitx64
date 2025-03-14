"""
Dual-Adversarial Validator
------------------------
ระบบตรวจสอบคุณภาพข้อมูลแบบ Dual-Adversarial Cascade
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from .quality_metrics import QualityMetrics
from .thai_specific_metrics import ThaiSpecificMetrics

class DualAdversarialValidator:
    """ระบบตรวจสอบคุณภาพข้อมูลแบบคู่ขนานที่แข่งขันกัน"""

    def __init__(self, ai_service_manager):
        """
        เริ่มต้นตัวตรวจสอบแบบ Dual-Adversarial
        
        Args:
            ai_service_manager: ตัวจัดการ AI service สำหรับใช้โมเดลต่างๆ
        """
        self.ai_service = ai_service_manager
        self.quality_metrics = QualityMetrics()
        self.thai_metrics = ThaiSpecificMetrics()
        
        # System A prompt template สำหรับการสร้างข้อมูล
        self.generator_prompt = """
        คุณเป็นระบบ A ในการตรวจสอบคุณภาพข้อมูลแบบ Dual-Adversarial
        กรุณาตรวจสอบความถูกต้องและคุณภาพของข้อมูลต่อไปนี้:

        {data}

        โดยพิจารณา:
        1. ความถูกต้องทางภาษา
        2. ความสมเหตุสมผลของเนื้อหา
        3. ความสอดคล้องของข้อมูล
        4. รูปแบบตามประเภทงาน ({task_type})

        กรุณาระบุ:
        1. ปัญหาที่พบ (ถ้ามี)
        2. คะแนนคุณภาพ (0-1)
        3. คำแนะนำในการปรับปรุง
        """
        
        # System B prompt template สำหรับการตรวจจับข้อผิดพลาด
        self.critic_prompt = """
        คุณเป็นระบบ B ในการตรวจสอบคุณภาพข้อมูลแบบ Dual-Adversarial
        กรุณาวิเคราะห์และระบุข้อผิดพลาดในข้อมูลต่อไปนี้:

        {data}

        มองหา:
        1. ข้อผิดพลาดทางตรรกะ
        2. ความไม่สอดคล้อง
        3. ความไม่สมเหตุสมผล
        4. ปัญหาเฉพาะของงานประเภท {task_type}

        ระบุ:
        1. ข้อผิดพลาดที่พบ
        2. ระดับความรุนแรง (1-5)
        3. ผลกระทบต่อคุณภาพข้อมูล
        """

    async def validate_samples(self, samples: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """
        ตรวจสอบตัวอย่างข้อมูลด้วยระบบ Dual-Adversarial
        
        Args:
            samples: รายการตัวอย่างข้อมูล
            task_type: ประเภทของงาน
            
        Returns:
            Dict[str, Any]: ผลการตรวจสอบและคำแนะนำ
        """
        results = []
        
        for idx, sample in enumerate(samples):
            # System A: ตรวจสอบคุณภาพและสร้างข้อเสนอแนะ
            generator_results = await self._run_generator_check(sample, task_type)
            
            # System B: ตรวจจับข้อผิดพลาดและปัญหา
            critic_results = await self._run_critic_check(sample, task_type)
            
            # ผสานผลการตรวจสอบ
            combined_score = self._combine_scores(generator_results, critic_results)
            
            results.append({
                "index": idx,
                "sample": sample,
                "generator_feedback": generator_results,
                "critic_feedback": critic_results,
                "combined_score": combined_score,
                "needs_revision": combined_score < 0.8
            })
        
        # วิเคราะห์คุณภาพรวม
        overall_metrics = self._calculate_overall_metrics(results)
        
        return {
            "results": results,
            "overall_metrics": overall_metrics,
            "recommendations": self._generate_recommendations(overall_metrics)
        }

    async def _run_generator_check(self, sample: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """ตรวจสอบด้วย System A (Generator)"""
        prompt = self.generator_prompt.format(
            data=json.dumps(sample, ensure_ascii=False, indent=2),
            task_type=task_type
        )
        
        try:
            response = await self.ai_service.generate_with_thai_model(prompt)
            return self._parse_generator_response(response)
        except Exception as e:
            return {
                "error": f"เกิดข้อผิดพลาดในการตรวจสอบ: {str(e)}",
                "quality_score": 0.0,
                "suggestions": []
            }

    async def _run_critic_check(self, sample: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """ตรวจสอบด้วย System B (Critic)"""
        prompt = self.critic_prompt.format(
            data=json.dumps(sample, ensure_ascii=False, indent=2),
            task_type=task_type
        )
        
        try:
            response = await self.ai_service.mistral.generate_text(prompt)
            return self._parse_critic_response(response)
        except Exception as e:
            return {
                "error": f"เกิดข้อผิดพลาดในการตรวจจับ: {str(e)}",
                "issues": [],
                "severity_score": 5.0
            }

    def _combine_scores(self, generator_results: Dict[str, Any], critic_results: Dict[str, Any]) -> float:
        """รวมคะแนนจากทั้งสองระบบ"""
        generator_score = generator_results.get("quality_score", 0.0)
        critic_severity = critic_results.get("severity_score", 5.0)
        
        # แปลงความรุนแรงของปัญหา (1-5) เป็นคะแนน (0-1)
        critic_score = 1.0 - (critic_severity - 1) / 4
        
        # ให้น้ำหนักคะแนนจากทั้งสองระบบ
        return 0.6 * generator_score + 0.4 * critic_score

    def _calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """คำนวณเมตริกรวมของชุดข้อมูล"""
        scores = [r["combined_score"] for r in results]
        revision_needed = sum(1 for r in results if r["needs_revision"])
        
        return {
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "std_score": self._calculate_std(scores),
            "samples_needing_revision": revision_needed,
            "revision_ratio": revision_needed / len(results) if results else 0.0
        }

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """สร้างคำแนะนำจากผลการวิเคราะห์"""
        recommendations = []
        
        if metrics["average_score"] < 0.8:
            recommendations.append("ควรปรับปรุงคุณภาพข้อมูลโดยรวม")
            
        if metrics["revision_ratio"] > 0.2:
            recommendations.append(
                f"พบข้อมูลที่ต้องแก้ไข {metrics['samples_needing_revision']} รายการ "
                f"({metrics['revision_ratio']*100:.1f}%) ควรทบทวนกระบวนการสร้างข้อมูล"
            )
            
        if metrics["std_score"] > 0.2:
            recommendations.append("ความแปรปรวนของคุณภาพข้อมูลสูงเกินไป ควรสร้างมาตรฐานการตรวจสอบที่เข้มงวดขึ้น")
            
        return recommendations

    def _calculate_std(self, values: List[float]) -> float:
        """คำนวณส่วนเบี่ยงเบนมาตรฐาน"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return (squared_diff_sum / len(values)) ** 0.5

    def _parse_generator_response(self, response: str) -> Dict[str, Any]:
        """แยกวิเคราะห์ผลตอบกลับจาก Generator"""
        try:
            # แยกวิเคราะห์ผลตอบกลับจาก LLM (ตัวอย่างการแยกอย่างง่าย)
            lines = response.split('\n')
            quality_score = 0.0
            suggestions = []
            
            for line in lines:
                if "คะแนน" in line and ":" in line:
                    try:
                        score_text = line.split(":")[1].strip()
                        quality_score = float(score_text)
                    except:
                        pass
                elif "แนะนำ" in line or "ปรับปรุง" in line:
                    suggestions.append(line.strip())
            
            return {
                "quality_score": quality_score,
                "suggestions": suggestions,
                "raw_response": response
            }
        except Exception as e:
            return {
                "error": f"ไม่สามารถแยกวิเคราะห์ผลตอบกลับ: {str(e)}",
                "quality_score": 0.0,
                "suggestions": [],
                "raw_response": response
            }

    def _parse_critic_response(self, response: str) -> Dict[str, Any]:
        """แยกวิเคราะห์ผลตอบกลับจาก Critic"""
        try:
            lines = response.split('\n')
            issues = []
            severity_score = 1.0  # ค่าเริ่มต้น (ดีที่สุด)
            
            for line in lines:
                if "ความรุนแรง" in line and ":" in line:
                    try:
                        severity_text = line.split(":")[1].strip()
                        severity_score = float(severity_text)
                    except:
                        pass
                elif "ปัญหา" in line or "ข้อผิดพลาด" in line:
                    issues.append(line.strip())
            
            return {
                "issues": issues,
                "severity_score": severity_score,
                "raw_response": response
            }
        except Exception as e:
            return {
                "error": f"ไม่สามารถแยกวิเคราะห์ผลตอบกลับ: {str(e)}",
                "issues": [],
                "severity_score": 5.0,  # ค่าแย่ที่สุดเมื่อเกิดข้อผิดพลาด
                "raw_response": response
            }
