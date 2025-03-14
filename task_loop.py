"""
Task Loop Manager
---------------
ระบบจัดการลำดับงานสำหรับการสร้างชุดข้อมูล NLP พร้อมระบบตรวจสอบคุณภาพ
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from dataset_generator import ThaiNLPDatasetGenerator
from data_quality.dataset_quality_pipeline import QualityConfig

@dataclass
class TaskConfig:
    """การกำหนดค่าสำหรับงาน"""
    task_type: str
    num_samples: int
    min_quality_score: float = 0.8
    max_retries: int = 3
    params: Dict[str, Any] = None
    enable_semantic_expansion: bool = True

@dataclass
class TaskResult:
    """ผลลัพธ์ของงาน"""
    task_id: str
    start_time: str
    end_time: str
    status: str
    quality_score: float
    dataset_path: str
    error: Optional[str] = None
    recommendations: List[str] = None

class TaskLoop:
    """ระบบจัดการลำดับงานพร้อมการตรวจสอบคุณภาพ"""

    def __init__(
        self, 
        output_dir: str = "datasets",
        checkpoint_file: str = "task_checkpoint.json",
        quality_config: Optional[QualityConfig] = None
    ):
        self.output_dir = output_dir
        self.checkpoint_file = checkpoint_file
        self.quality_config = quality_config or QualityConfig(
            min_quality_score=0.8,
            semantic_expansion_depth=3,
            enable_dual_validation=True,
            enable_thai_specific=True,
            enable_semantic_tree=True
        )
        
        # สร้างโฟลเดอร์ output ถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        
        # โหลด checkpoint ถ้ามี
        self.completed_tasks = self._load_checkpoint()
        
        # สร้าง dataset generator
        self.generator = ThaiNLPDatasetGenerator(
            output_dir=output_dir,
            quality_config=self.quality_config
        )

    def _load_checkpoint(self) -> Dict[str, TaskResult]:
        """โหลดข้อมูล checkpoint"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    task_id: TaskResult(**task_data)
                    for task_id, task_data in data.items()
                }
        return {}

    def _save_checkpoint(self):
        """บันทึก checkpoint"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    task_id: asdict(result)
                    for task_id, result in self.completed_tasks.items()
                },
                f,
                ensure_ascii=False,
                indent=2
            )

    def _generate_task_id(self, task_config: TaskConfig) -> str:
        """สร้าง task ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{task_config.task_type}_{timestamp}"

    async def run_task(self, task_config: TaskConfig) -> TaskResult:
        """
        รันงานหนึ่งงานพร้อมการตรวจสอบคุณภาพ
        
        Args:
            task_config: การกำหนดค่าของงาน
            
        Returns:
            TaskResult: ผลลัพธ์ของงาน
        """
        task_id = self._generate_task_id(task_config)
        start_time = datetime.now().isoformat()
        
        try:
            # เรียกใช้งาน generator function ตามประเภทงาน
            generator_func = getattr(self.generator, f"generate_{task_config.task_type}")
            params = task_config.params or {}
            
            # สร้างชุดข้อมูลและตรวจสอบคุณภาพ
            attempts = 0
            while attempts < task_config.max_retries:
                try:
                    dataset = await generator_func(
                        num_samples=task_config.num_samples,
                        **params
                    )
                    
                    # บันทึกและตรวจสอบคุณภาพ
                    quality_results = await self.generator.save_dataset(
                        dataset, 
                        f"{task_config.task_type}_thai", 
                        task_config.task_type
                    )
                    
                    quality_score = quality_results["summary"]["overall_quality_score"]
                    
                    if quality_score >= task_config.min_quality_score:
                        result = TaskResult(
                            task_id=task_id,
                            start_time=start_time,
                            end_time=datetime.now().isoformat(),
                            status="success",
                            quality_score=quality_score,
                            dataset_path=os.path.join(self.output_dir, f"{task_config.task_type}_thai"),
                            recommendations=quality_results["recommendations"]
                        )
                        break
                    else:
                        print(f"คุณภาพต่ำกว่าเกณฑ์ ({quality_score:.2f} < {task_config.min_quality_score}) ลองใหม่...")
                        attempts += 1
                        
                except Exception as e:
                    print(f"เกิดข้อผิดพลาด: {str(e)}")
                    attempts += 1
                    
            if attempts == task_config.max_retries:
                result = TaskResult(
                    task_id=task_id,
                    start_time=start_time,
                    end_time=datetime.now().isoformat(),
                    status="failed",
                    quality_score=0.0,
                    dataset_path="",
                    error=f"ไม่สามารถสร้างชุดข้อมูลที่มีคุณภาพตามเกณฑ์ได้หลังจากลองแล้ว {attempts} ครั้ง"
                )
                
        except Exception as e:
            result = TaskResult(
                task_id=task_id,
                start_time=start_time,
                end_time=datetime.now().isoformat(),
                status="error",
                quality_score=0.0,
                dataset_path="",
                error=str(e)
            )
            
        # บันทึกผลลัพธ์
        self.completed_tasks[task_id] = result
        self._save_checkpoint()
        
        return result

    async def run_tasks(self, tasks: List[TaskConfig]) -> Dict[str, TaskResult]:
        """
        รันหลายงานพร้อมกัน
        
        Args:
            tasks: รายการงานที่จะรัน
            
        Returns:
            Dict[str, TaskResult]: ผลลัพธ์ของแต่ละงาน
        """
        results = {}
        for task in tasks:
            result = await self.run_task(task)
            results[result.task_id] = result
        return results

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """ดูสถานะของงาน"""
        return self.completed_tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskResult]:
        """ดูสถานะของทุกงาน"""
        return self.completed_tasks

def create_example_tasks() -> List[TaskConfig]:
    """สร้างตัวอย่างงาน"""
    return [
        TaskConfig(
            task_type="text_classification",
            num_samples=100,
            params={"num_classes": 3}
        ),
        TaskConfig(
            task_type="token_classification",
            num_samples=100,
            params={"entity_types": ["PER", "ORG", "LOC"]}
        ),
        TaskConfig(
            task_type="question_answering",
            num_samples=50
        ),
        TaskConfig(
            task_type="summarization",
            num_samples=50
        ),
        TaskConfig(
            task_type="translation",
            num_samples=100,
            params={"language_pairs": [("en", "th"), ("th", "en")]}
        )
    ]

async def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    # สร้าง task loop
    task_loop = TaskLoop()
    
    # รันตัวอย่างงาน
    tasks = create_example_tasks()
    results = await task_loop.run_tasks(tasks)
    
    # แสดงผลลัพธ์
    print("\nผลการทำงาน:")
    for task_id, result in results.items():
        print(f"\nTask: {task_id}")
        print(f"Status: {result.status}")
        print(f"Quality Score: {result.quality_score:.2f}")
        if result.error:
            print(f"Error: {result.error}")
        if result.recommendations:
            print("Recommendations:")
            for rec in result.recommendations:
                print(f"- {rec}")

if __name__ == "__main__":
    asyncio.run(main())
