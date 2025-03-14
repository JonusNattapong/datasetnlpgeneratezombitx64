"""
Recursive Semantic Tree Expansion
------------------------------
ระบบขยายชุดข้อมูลแบบรุกขมรรค (tree) ที่ขยายตัวตามความหมาย
"""

from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import json
from dataclasses import dataclass
from enum import Enum

class RelationType(Enum):
    """ประเภทความสัมพันธ์ระหว่างโหนด"""
    IS_A = "is_a"                   # เป็นประเภทของ
    PART_OF = "part_of"            # เป็นส่วนหนึ่งของ
    HAS_PROPERTY = "has_property"  # มีคุณสมบัติ
    RELATED_TO = "related_to"      # เกี่ยวข้องกับ
    SIMILAR_TO = "similar_to"      # คล้ายคลึงกับ
    OPPOSITE_OF = "opposite_of"    # ตรงข้ามกับ

@dataclass
class SemanticNode:
    """โหนดในต้นไม้ความหมาย"""
    concept: str                          # แนวคิดหรือคำหลัก
    examples: List[str]                   # ตัวอย่างข้อความ
    relations: Dict[RelationType, List[str]]  # ความสัมพันธ์กับโหนดอื่น
    metadata: Dict[str, Any]              # ข้อมูลเพิ่มเติม
    depth: int = 0                        # ระดับความลึกในต้นไม้

class SemanticTreeExpander:
    """ระบบขยายชุดข้อมูลแบบรุกขมรรค"""
    
    def __init__(self, ai_service_manager=None):
        """
        เริ่มต้นระบบขยายต้นไม้ความหมาย
        
        Args:
            ai_service_manager: ตัวจัดการ AI service สำหรับการขยายความหมาย
        """
        self.ai_service = ai_service_manager
        self.nodes: Dict[str, SemanticNode] = {}
        self.expansion_prompt = """
        กรุณาวิเคราะห์แนวคิด "{concept}" และระบุ:
        1. ความสัมพันธ์แบบ "เป็นประเภทของ"
        2. ความสัมพันธ์แบบ "เป็นส่วนหนึ่งของ"
        3. คุณสมบัติที่สำคัญ
        4. แนวคิดที่เกี่ยวข้อง
        5. แนวคิดที่คล้ายคลึง
        6. แนวคิดที่ตรงข้าม
        
        พร้อมตัวอย่างประโยคที่เกี่ยวข้องกับแต่ละความสัมพันธ์
        """
        
    def add_node(self, concept: str, examples: List[str], depth: int = 0) -> SemanticNode:
        """เพิ่มโหนดใหม่ในต้นไม้"""
        if concept not in self.nodes:
            self.nodes[concept] = SemanticNode(
                concept=concept,
                examples=examples,
                relations={rel_type: [] for rel_type in RelationType},
                metadata={},
                depth=depth
            )
        return self.nodes[concept]
    
    def add_relation(self, source: str, target: str, relation_type: RelationType):
        """เพิ่มความสัมพันธ์ระหว่างโหนด"""
        if source in self.nodes and target in self.nodes:
            if target not in self.nodes[source].relations[relation_type]:
                self.nodes[source].relations[relation_type].append(target)
    
    async def expand_concept(self, concept: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        ขยายแนวคิดแบบรุกขมรรค
        
        Args:
            concept: แนวคิดเริ่มต้น
            max_depth: ความลึกสูงสุดในการขยาย
            
        Returns:
            Dict[str, Any]: ผลการขยายแนวคิด
        """
        if not self.ai_service:
            raise ValueError("ต้องกำหนด AI service manager สำหรับการขยายแนวคิด")
            
        # เริ่มจากแนวคิดหลัก
        visited = set()
        await self._expand_node(concept, 0, max_depth, visited)
        
        # สร้างกราฟความสัมพันธ์
        graph = self._build_relation_graph()
        
        # วิเคราะห์ความครอบคลุม
        coverage = self._analyze_coverage()
        
        return {
            "nodes": self.nodes,
            "graph": graph,
            "coverage": coverage
        }
    
    async def _expand_node(self, concept: str, current_depth: int, max_depth: int, visited: Set[str]):
        """ขยายโหนดแบบรุกขมรรค"""
        if current_depth >= max_depth or concept in visited:
            return
            
        visited.add(concept)
        
        # สร้างคำขอไปยัง AI service
        prompt = self.expansion_prompt.format(concept=concept)
        try:
            response = await self.ai_service.generate_with_thai_model(prompt)
            expansion = self._parse_expansion_response(response)
            
            # สร้างโหนดใหม่และความสัมพันธ์
            for relation_type, concepts in expansion["relations"].items():
                for related_concept in concepts:
                    # เพิ่มโหนดใหม่ถ้ายังไม่มี
                    if related_concept not in self.nodes:
                        self.add_node(
                            related_concept,
                            expansion["examples"].get(related_concept, []),
                            current_depth + 1
                        )
                    # เพิ่มความสัมพันธ์
                    self.add_relation(concept, related_concept, RelationType[relation_type])
                    
                    # ขยายต่อแบบรุกขมรรค
                    await self._expand_node(related_concept, current_depth + 1, max_depth, visited)
                    
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการขยายแนวคิด '{concept}': {str(e)}")
    
    def _parse_expansion_response(self, response: str) -> Dict[str, Any]:
        """แยกวิเคราะห์ผลตอบกลับจาก AI"""
        # TODO: พัฒนาการแยกวิเคราะห์ที่ซับซ้อนขึ้น
        # ตัวอย่างการแยกวิเคราะห์อย่างง่าย
        expansion = {
            "relations": defaultdict(list),
            "examples": defaultdict(list)
        }
        
        lines = response.split("\n")
        current_type = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # ตรวจจับประเภทความสัมพันธ์
            for rel_type in RelationType:
                if rel_type.value in line.lower():
                    current_type = rel_type
                    break
            
            # เพิ่มแนวคิดและตัวอย่าง
            if current_type and ":" in line:
                concept = line.split(":", 1)[1].strip()
                if concept:
                    expansion["relations"][current_type.name].append(concept)
        
        return expansion
    
    def _build_relation_graph(self) -> Dict[str, Any]:
        """สร้างกราฟความสัมพันธ์"""
        graph = {
            "nodes": [],
            "edges": []
        }
        
        # เพิ่มโหนด
        for concept, node in self.nodes.items():
            graph["nodes"].append({
                "id": concept,
                "depth": node.depth,
                "example_count": len(node.examples)
            })
            
        # เพิ่มเส้นเชื่อม
        for source, node in self.nodes.items():
            for rel_type, targets in node.relations.items():
                for target in targets:
                    graph["edges"].append({
                        "source": source,
                        "target": target,
                        "type": rel_type.value
                    })
        
        return graph
    
    def _analyze_coverage(self) -> Dict[str, Any]:
        """วิเคราะห์ความครอบคลุมของต้นไม้"""
        total_nodes = len(self.nodes)
        total_relations = sum(
            len(rels) for node in self.nodes.values()
            for rels in node.relations.values()
        )
        total_examples = sum(
            len(node.examples) for node in self.nodes.values()
        )
        
        # คำนวณความหลากหลายของความสัมพันธ์
        relation_type_counts = defaultdict(int)
        for node in self.nodes.values():
            for rel_type, targets in node.relations.items():
                relation_type_counts[rel_type.value] += len(targets)
        
        return {
            "total_nodes": total_nodes,
            "total_relations": total_relations,
            "total_examples": total_examples,
            "relation_distribution": dict(relation_type_counts),
            "average_relations_per_node": total_relations / total_nodes if total_nodes > 0 else 0,
            "average_examples_per_node": total_examples / total_nodes if total_nodes > 0 else 0
        }

    def export_to_json(self, filepath: str):
        """ส่งออกต้นไม้ความหมายเป็น JSON"""
        tree_data = {
            "nodes": {
                concept: {
                    "concept": node.concept,
                    "examples": node.examples,
                    "relations": {
                        rel_type.value: targets 
                        for rel_type, targets in node.relations.items()
                    },
                    "metadata": node.metadata,
                    "depth": node.depth
                }
                for concept, node in self.nodes.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str, ai_service_manager=None):
        """โหลดต้นไม้ความหมายจาก JSON"""
        expander = cls(ai_service_manager)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        for concept, node_data in tree_data["nodes"].items():
            node = expander.add_node(
                concept=node_data["concept"],
                examples=node_data["examples"],
                depth=node_data["depth"]
            )
            
            # โหลดความสัมพันธ์
            for rel_type_str, targets in node_data["relations"].items():
                try:
                    rel_type = RelationType(rel_type_str)
                    for target in targets:
                        expander.add_relation(concept, target, rel_type)
                except ValueError:
                    print(f"ข้ามความสัมพันธ์ที่ไม่รู้จัก: {rel_type_str}")
            
            # โหลด metadata
            node.metadata.update(node_data.get("metadata", {}))
        
        return expander
