# -*- coding: utf-8 -*-
"""
医学知识图谱RAG - 独立模块（不集成到主系统）
功能: 实体抽取、关系抽取、图谱构建、图谱检索
依赖: neo4j, spacy (可选scispacy)

使用方法:
1. 安装依赖: pip install neo4j spacy
2. 启动Neo4j: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
3. 运行: python src/graph/medical_graph_rag.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

# 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


@dataclass
class Entity:
    """医学实体"""
    text: str
    type: str  # DISEASE, DRUG, SYMPTOM, TREATMENT, GENE, ANATOMY
    start: int = -1
    end: int = -1


@dataclass
class Relation:
    """实体关系"""
    source: str
    target: str
    relation: str  # TREATS, CAUSES, PREVENTS, ASSOCIATED_WITH, DIAGNOSES


class MedicalEntityExtractor:
    """医学实体抽取器"""
    
    def __init__(self, use_scispacy: bool = False):
        """
        初始化实体抽取器
        
        Args:
            use_scispacy: 是否使用scispaCy医学模型
        """
        self.nlp = None
        self.use_scispacy = use_scispacy
        
        # 医学实体关键词（简单规则匹配）
        self.disease_keywords = [
            "diabetes", "cancer", "hypertension", "stroke", "pneumonia",
            "asthma", "arthritis", "alzheimer", "parkinson", "epilepsy",
            "tuberculosis", "hepatitis", "cirrhosis", "fibrosis", "carcinoma"
        ]
        self.drug_keywords = [
            "insulin", "metformin", "aspirin", "statin", "antibiotic",
            "chemotherapy", "immunotherapy", "vaccine", "inhibitor", "blocker"
        ]
        self.symptom_keywords = [
            "pain", "fever", "fatigue", "nausea", "headache", "cough",
            "inflammation", "swelling", "bleeding", "dysfunction"
        ]
        
        self._init_nlp()
    
    def _init_nlp(self):
        """初始化NLP模型"""
        try:
            import spacy
            if self.use_scispacy:
                try:
                    self.nlp = spacy.load("en_core_sci_sm")
                    print("✅ 加载scispaCy医学模型")
                except:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("⚠️ scispaCy未安装，使用通用模型")
            else:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ 加载spaCy通用模型")
        except Exception as e:
            print(f"⚠️ spaCy加载失败: {e}，使用规则匹配")
            self.nlp = None
    
    def extract_entities(self, text: str, mesh_terms: List[str] = None) -> List[Entity]:
        """
        从文本中抽取医学实体
        
        Args:
            text: 输入文本
            mesh_terms: MeSH主题词（来自PubMed）
            
        Returns:
            实体列表
        """
        entities = []
        text_lower = text.lower()
        
        # 1. 使用NLP模型抽取
        if self.nlp:
            doc = self.nlp(text[:5000])  # 限制长度
            for ent in doc.ents:
                entity_type = self._classify_entity(ent.text, ent.label_)
                if entity_type:
                    entities.append(Entity(
                        text=ent.text,
                        type=entity_type,
                        start=ent.start_char,
                        end=ent.end_char
                    ))
        
        # 2. 规则匹配补充
        for keyword in self.disease_keywords:
            if keyword in text_lower:
                entities.append(Entity(text=keyword, type="DISEASE"))
        
        for keyword in self.drug_keywords:
            if keyword in text_lower:
                entities.append(Entity(text=keyword, type="DRUG"))
        
        for keyword in self.symptom_keywords:
            if keyword in text_lower:
                entities.append(Entity(text=keyword, type="SYMPTOM"))
        
        # 3. 使用MeSH Terms
        if mesh_terms:
            for term in mesh_terms:
                entity_type = self._classify_mesh_term(term)
                entities.append(Entity(text=term, type=entity_type))
        
        # 4. 去重
        seen = set()
        unique_entities = []
        for ent in entities:
            key = (ent.text.lower(), ent.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        return unique_entities
    
    def _classify_entity(self, text: str, spacy_label: str) -> Optional[str]:
        """分类实体类型"""
        text_lower = text.lower()
        
        # 基于spaCy标签
        label_mapping = {
            "DISEASE": "DISEASE",
            "CHEMICAL": "DRUG",
            "GENE": "GENE",
            "PROTEIN": "GENE",
            "CELL_TYPE": "ANATOMY",
            "ORGAN": "ANATOMY"
        }
        if spacy_label in label_mapping:
            return label_mapping[spacy_label]
        
        # 基于关键词
        if any(kw in text_lower for kw in self.disease_keywords):
            return "DISEASE"
        if any(kw in text_lower for kw in self.drug_keywords):
            return "DRUG"
        if any(kw in text_lower for kw in self.symptom_keywords):
            return "SYMPTOM"
        
        return None
    
    def _classify_mesh_term(self, term: str) -> str:
        """分类MeSH术语"""
        term_lower = term.lower()
        if any(kw in term_lower for kw in ["disease", "syndrome", "disorder", "cancer", "oma"]):
            return "DISEASE"
        if any(kw in term_lower for kw in ["drug", "therapy", "treatment", "agent"]):
            return "DRUG"
        return "CONCEPT"


class MedicalRelationExtractor:
    """医学关系抽取器"""
    
    def __init__(self, llm_client=None):
        """
        初始化关系抽取器
        
        Args:
            llm_client: OpenAI兼容的LLM客户端（可选）
        """
        self.llm_client = llm_client
        
        # 关系模式（简单规则）
        self.relation_patterns = [
            (r"(\w+)\s+(?:treats?|treatment for)\s+(\w+)", "TREATS"),
            (r"(\w+)\s+(?:causes?|leads? to)\s+(\w+)", "CAUSES"),
            (r"(\w+)\s+(?:prevents?|prevention of)\s+(\w+)", "PREVENTS"),
            (r"(\w+)\s+(?:associated with|related to)\s+(\w+)", "ASSOCIATED_WITH"),
            (r"(\w+)\s+(?:diagnoses?|diagnosis of)\s+(\w+)", "DIAGNOSES"),
        ]
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        从文本中抽取实体间关系
        
        Args:
            text: 输入文本
            entities: 已抽取的实体列表
            
        Returns:
            关系列表
        """
        relations = []
        
        # 1. 规则匹配
        for pattern, rel_type in self.relation_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if len(match) == 2:
                    relations.append(Relation(
                        source=match[0],
                        target=match[1],
                        relation=rel_type
                    ))
        
        # 2. 基于实体共现推断关系
        entity_texts = [e.text.lower() for e in entities]
        drug_entities = [e for e in entities if e.type == "DRUG"]
        disease_entities = [e for e in entities if e.type == "DISEASE"]
        
        # 药物-疾病共现 -> TREATS关系
        for drug in drug_entities:
            for disease in disease_entities:
                relations.append(Relation(
                    source=drug.text,
                    target=disease.text,
                    relation="TREATS"
                ))
        
        # 3. 使用LLM抽取（如果可用）
        if self.llm_client and len(entities) >= 2:
            llm_relations = self._extract_with_llm(text, entities)
            relations.extend(llm_relations)
        
        # 去重
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel.source.lower(), rel.relation, rel.target.lower())
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def _extract_with_llm(self, text: str, entities: List[Entity]) -> List[Relation]:
        """使用LLM抽取关系"""
        if not self.llm_client:
            return []
        
        entity_list = "\n".join([f"- {e.text} ({e.type})" for e in entities[:15]])
        
        prompt = f"""Extract medical relationships from the text.

Text: {text[:1500]}

Entities:
{entity_list}

Output JSON array of relationships:
[{{"source": "entity1", "relation": "TREATS|CAUSES|PREVENTS|ASSOCIATED_WITH", "target": "entity2"}}]

Only extract relationships explicitly mentioned. Return valid JSON."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [Relation(**r) for r in data]
        except Exception as e:
            print(f"LLM关系抽取失败: {e}")
        
        return []


class MedicalKnowledgeGraph:
    """医学知识图谱"""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, 
                 password: str = NEO4J_PASSWORD):
        """初始化知识图谱"""
        self.driver = None
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print("✅ Neo4j连接成功")
        except Exception as e:
            print(f"⚠️ Neo4j连接失败: {e}")
    
    def add_entity(self, entity: Entity, source_pmid: str = None):
        """添加实体节点"""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            session.run(
                """
                MERGE (n:Entity {id: $id})
                SET n.text = $text, n.type = $type, n.source_pmid = $source_pmid
                """,
                id=entity.text.lower().replace(" ", "_"),
                text=entity.text,
                type=entity.type,
                source_pmid=source_pmid
            )
    
    def add_relation(self, relation: Relation, source_pmid: str = None):
        """添加关系边"""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            # 动态创建关系类型
            query = f"""
            MATCH (a:Entity {{id: $source_id}})
            MATCH (b:Entity {{id: $target_id}})
            MERGE (a)-[r:{relation.relation}]->(b)
            SET r.source_pmid = $source_pmid
            """
            session.run(
                query,
                source_id=relation.source.lower().replace(" ", "_"),
                target_id=relation.target.lower().replace(" ", "_"),
                source_pmid=source_pmid
            )
    
    def query_disease_treatments(self, disease: str) -> List[Dict]:
        """查询疾病的治疗方法"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Entity)-[:TREATS]-(t:Entity)
                WHERE toLower(d.text) CONTAINS toLower($disease)
                RETURN DISTINCT t.text as treatment, t.type as type
                LIMIT 20
                """,
                disease=disease
            )
            return [dict(record) for record in result]
    
    def query_related_entities(self, entity: str, max_hops: int = 2) -> List[Dict]:
        """查询相关实体（多跳）"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (start:Entity)-[*1..{max_hops}]-(end:Entity)
                WHERE toLower(start.text) CONTAINS toLower($entity)
                RETURN DISTINCT end.text as entity, end.type as type,
                       length(path) as distance
                ORDER BY distance
                LIMIT 30
                """,
                entity=entity
            )
            return [dict(record) for record in result]
    
    def get_statistics(self) -> Dict:
        """获取图谱统计"""
        if not self.driver:
            return {}
        
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            return {
                "nodes": node_count,
                "edges": edge_count
            }
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()


class GraphRAGRetriever:
    """GraphRAG检索器 - 结合图谱和向量检索"""
    
    def __init__(self, knowledge_graph: MedicalKnowledgeGraph,
                 entity_extractor: MedicalEntityExtractor):
        self.kg = knowledge_graph
        self.extractor = entity_extractor
    
    def retrieve(self, query: str, top_k: int = 10) -> Dict:
        """
        图谱增强检索
        
        Args:
            query: 用户查询
            top_k: 返回数量
            
        Returns:
            检索结果
        """
        # 1. 从查询中抽取实体
        entities = self.extractor.extract_entities(query)
        
        if not entities:
            return {"entities": [], "graph_results": [], "context": ""}
        
        # 2. 图谱检索
        graph_results = []
        for entity in entities:
            # 查询相关实体
            related = self.kg.query_related_entities(entity.text, max_hops=2)
            graph_results.extend(related)
            
            # 如果是疾病，查询治疗方法
            if entity.type == "DISEASE":
                treatments = self.kg.query_disease_treatments(entity.text)
                graph_results.extend(treatments)
        
        # 3. 去重并排序
        seen = set()
        unique_results = []
        for r in graph_results:
            key = r.get("entity") or r.get("treatment")
            if key and key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        # 4. 生成上下文
        context = self._generate_context(entities, unique_results[:top_k])
        
        return {
            "query": query,
            "entities": [{"text": e.text, "type": e.type} for e in entities],
            "graph_results": unique_results[:top_k],
            "context": context
        }
    
    def _generate_context(self, entities: List[Entity], results: List[Dict]) -> str:
        """生成图谱上下文"""
        lines = []
        
        # 实体信息
        entity_info = ", ".join([f"{e.text}({e.type})" for e in entities])
        lines.append(f"识别的医学实体: {entity_info}")
        
        # 图谱关系
        if results:
            lines.append("\n相关知识图谱信息:")
            for r in results[:10]:
                if "treatment" in r:
                    lines.append(f"  - 治疗方法: {r['treatment']}")
                elif "entity" in r:
                    lines.append(f"  - 相关实体: {r['entity']} ({r.get('type', 'N/A')})")
        
        return "\n".join(lines)


def build_graph_from_articles(articles: List[Dict], max_articles: int = 1000):
    """
    从文章列表构建知识图谱
    
    Args:
        articles: 文章列表
        max_articles: 最大处理文章数
    """
    print(f"开始构建知识图谱，处理 {min(len(articles), max_articles)} 篇文章...")
    
    # 初始化组件
    extractor = MedicalEntityExtractor(use_scispacy=False)
    relation_extractor = MedicalRelationExtractor()
    kg = MedicalKnowledgeGraph()
    
    if not kg.driver:
        print("❌ Neo4j未连接，无法构建图谱")
        return
    
    # 处理文章
    for i, article in enumerate(articles[:max_articles]):
        if (i + 1) % 100 == 0:
            print(f"进度: {i+1}/{max_articles}")
        
        pmid = article.get("pmid", "")
        text = f"{article.get('title', '')} {article.get('abstract', '')}"
        mesh_terms = article.get("mesh_terms", [])
        
        # 抽取实体
        entities = extractor.extract_entities(text, mesh_terms)
        
        # 抽取关系
        relations = relation_extractor.extract_relations(text, entities)
        
        # 添加到图谱
        for entity in entities:
            kg.add_entity(entity, pmid)
        
        for relation in relations:
            kg.add_relation(relation, pmid)
    
    # 统计
    stats = kg.get_statistics()
    print(f"\n✅ 图谱构建完成!")
    print(f"   节点数: {stats.get('nodes', 0)}")
    print(f"   边数: {stats.get('edges', 0)}")
    
    kg.close()


def demo():
    """演示GraphRAG功能"""
    print("=" * 60)
    print("医学知识图谱RAG演示")
    print("=" * 60)
    
    # 1. 实体抽取演示
    print("\n1. 实体抽取演示")
    extractor = MedicalEntityExtractor()
    
    test_text = """
    Type 2 diabetes mellitus is a chronic metabolic disorder characterized by 
    hyperglycemia. Metformin is the first-line treatment for type 2 diabetes.
    Patients may experience symptoms such as fatigue, increased thirst, and 
    frequent urination. Cardiovascular disease is a common complication.
    """
    
    entities = extractor.extract_entities(test_text)
    print(f"   抽取到 {len(entities)} 个实体:")
    for e in entities[:10]:
        print(f"   - {e.text} ({e.type})")
    
    # 2. 关系抽取演示
    print("\n2. 关系抽取演示")
    relation_extractor = MedicalRelationExtractor()
    relations = relation_extractor.extract_relations(test_text, entities)
    print(f"   抽取到 {len(relations)} 个关系:")
    for r in relations[:10]:
        print(f"   - {r.source} --[{r.relation}]--> {r.target}")
    
    # 3. 图谱检索演示（需要Neo4j）
    print("\n3. 图谱检索演示")
    kg = MedicalKnowledgeGraph()
    if kg.driver:
        # 添加测试数据
        for e in entities:
            kg.add_entity(e, "test")
        for r in relations:
            kg.add_relation(r, "test")
        
        # 查询
        retriever = GraphRAGRetriever(kg, extractor)
        result = retriever.retrieve("What are the treatments for diabetes?")
        
        print(f"   查询实体: {result['entities']}")
        print(f"   图谱结果: {len(result['graph_results'])} 条")
        print(f"   上下文:\n{result['context']}")
        
        kg.close()
    else:
        print("   ⚠️ Neo4j未连接，跳过图谱检索演示")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
