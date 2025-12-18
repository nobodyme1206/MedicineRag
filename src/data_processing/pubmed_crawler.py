# -*- coding: utf-8 -*-
"""
PubMed医学文献爬虫 
优化: 异步协程(aiohttp) + Session复用 + 高并发 + 批量写入
目标: 100个主题，每个主题50MB，总计5GB
"""

import asyncio
import aiohttp
import time
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import RAW_DATA_DIR, LOGS_DIR, PUBMED_EMAIL, PUBMED_API_KEY
from src.utils.logger import setup_logger

logger = setup_logger("pubmed_crawler", LOGS_DIR / "pubmed_crawler.log")

# 配置
TARGET_SIZE_MB = 50
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

# PubMed API配置
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE_URL}/efetch.fcgi"

# 100个医学主题
MEDICAL_TOPICS_EXPANDED = {
    # 代谢与内分泌 (10)
    "diabetes": ["diabetes mellitus", "diabetes type 2", "diabetes type 1", "diabetic complications", "insulin resistance"],
    "thyroid": ["thyroid disorders", "hypothyroidism", "hyperthyroidism", "thyroid nodule", "thyroid cancer"],
    "obesity": ["obesity metabolism", "obesity treatment", "bariatric surgery", "weight loss intervention"],
    "metabolic_syndrome": ["metabolic syndrome", "insulin sensitivity", "metabolic disorder"],
    "endocrine": ["adrenal insufficiency", "pituitary disorders", "cushing syndrome", "addison disease"],
    "lipid": ["hyperlipidemia treatment", "cholesterol management", "statin therapy", "dyslipidemia"],
    "bone_metabolism": ["osteoporosis treatment", "bone density", "calcium metabolism", "vitamin D deficiency"],
    "gout": ["gout hyperuricemia", "uric acid", "gout treatment", "crystal arthropathy"],
    "hormone": ["hormone replacement therapy", "growth hormone", "testosterone deficiency"],
    "nutrition": ["malnutrition", "nutritional deficiency", "enteral nutrition", "parenteral nutrition"],
    # 心血管 (15)
    "cardiovascular": ["cardiovascular disease", "heart disease prevention", "cardiac rehabilitation"],
    "hypertension": ["hypertension treatment", "blood pressure management", "antihypertensive therapy"],
    "heart_failure": ["heart failure", "congestive heart failure", "heart failure treatment", "cardiomyopathy"],
    "arrhythmia": ["atrial fibrillation", "cardiac arrhythmia", "ventricular tachycardia", "pacemaker"],
    "coronary": ["coronary artery disease", "angina pectoris", "coronary intervention", "stent placement"],
    "myocardial": ["acute myocardial infarction", "myocardial ischemia", "troponin", "STEMI NSTEMI"],
    "vascular": ["peripheral vascular disease", "arterial occlusion", "claudication", "angioplasty"],
    "aortic": ["aortic aneurysm", "aortic dissection", "aortic stenosis", "aortic valve"],
    "thrombosis": ["deep vein thrombosis", "pulmonary embolism", "anticoagulation therapy", "thromboprophylaxis"],
    "valve": ["valvular heart disease", "mitral regurgitation", "aortic regurgitation", "valve replacement"],
    "atherosclerosis": ["atherosclerosis", "plaque formation", "carotid stenosis", "endothelial dysfunction"],
    "congenital_heart": ["congenital heart disease", "atrial septal defect", "ventricular septal defect"],
    "cardiac_imaging": ["echocardiography", "cardiac MRI", "coronary angiography", "cardiac CT"],
    "lipid_cardio": ["cardiovascular risk", "lipid lowering", "PCSK9 inhibitor"],
    "heart_transplant": ["heart transplant", "mechanical circulatory support", "LVAD"],
    # 肿瘤 (15)
    "cancer_general": ["cancer treatment", "oncology", "tumor biology", "cancer immunotherapy"],
    "breast_cancer": ["breast cancer", "breast carcinoma", "mammography screening", "HER2 positive"],
    "lung_cancer": ["lung cancer", "non small cell lung cancer", "small cell lung cancer", "lung adenocarcinoma"],
    "prostate_cancer": ["prostate cancer", "PSA screening", "prostatectomy", "androgen deprivation"],
    "colorectal_cancer": ["colorectal cancer", "colon cancer", "rectal cancer", "colonoscopy screening"],
    "hematologic_cancer": ["leukemia lymphoma", "acute leukemia", "chronic leukemia", "non hodgkin lymphoma"],
    "skin_cancer": ["melanoma skin cancer", "basal cell carcinoma", "squamous cell carcinoma"],
    "brain_cancer": ["brain tumor glioma", "glioblastoma", "meningioma", "brain metastasis"],
    "pancreatic_cancer": ["pancreatic cancer", "pancreatic adenocarcinoma", "pancreatic neoplasm"],
    "gynecologic_cancer": ["ovarian cancer", "cervical cancer", "endometrial cancer", "uterine cancer"],
    "gastric_cancer": ["gastric cancer", "stomach cancer", "gastric adenocarcinoma", "H pylori cancer"],
    "liver_cancer": ["liver cancer hepatocellular", "hepatocellular carcinoma", "cholangiocarcinoma"],
    "kidney_cancer": ["kidney cancer renal", "renal cell carcinoma", "nephrectomy"],
    "bladder_cancer": ["bladder cancer", "urothelial carcinoma", "bladder tumor"],
    "thyroid_cancer": ["thyroid cancer", "papillary thyroid", "follicular thyroid", "thyroidectomy"],
    # 神经系统 (12)
    "alzheimer": ["alzheimer disease", "dementia", "cognitive impairment", "amyloid beta"],
    "parkinson": ["parkinson disease", "parkinsonian", "dopamine", "levodopa treatment"],
    "stroke": ["stroke cerebrovascular", "ischemic stroke", "hemorrhagic stroke", "stroke rehabilitation"],
    "epilepsy": ["epilepsy seizures", "seizure disorder", "antiepileptic drugs", "status epilepticus"],
    "multiple_sclerosis": ["multiple sclerosis", "demyelinating disease", "MS treatment", "interferon beta"],
    "headache": ["migraine headache", "tension headache", "cluster headache", "headache treatment"],
    "neuropathy": ["neuropathic pain", "peripheral neuropathy", "diabetic neuropathy", "nerve damage"],
    "als": ["amyotrophic lateral sclerosis", "motor neuron disease", "ALS treatment"],
    "movement_disorder": ["huntington disease", "dystonia", "tremor", "chorea"],
    "trauma_neuro": ["traumatic brain injury", "concussion", "brain trauma", "TBI rehabilitation"],
    "spinal": ["spinal cord injury", "paraplegia", "quadriplegia", "spinal rehabilitation"],
    "neurodegenerative": ["neurodegeneration", "frontotemporal dementia", "lewy body dementia"],
    # 呼吸系统 (10)
    "pneumonia": ["pneumonia", "community acquired pneumonia", "hospital acquired pneumonia", "aspiration pneumonia"],
    "asthma": ["asthma treatment", "bronchial asthma", "asthma exacerbation", "inhaled corticosteroid"],
    "copd": ["chronic obstructive pulmonary disease", "COPD exacerbation", "emphysema", "chronic bronchitis"],
    "pulmonary_fibrosis": ["pulmonary fibrosis", "idiopathic pulmonary fibrosis", "interstitial lung disease"],
    "cystic_fibrosis": ["cystic fibrosis", "CFTR", "cystic fibrosis treatment"],
    "tuberculosis": ["tuberculosis", "TB treatment", "multidrug resistant TB", "latent TB"],
    "respiratory_infection": ["respiratory infection", "bronchitis", "respiratory syncytial virus"],
    "sleep_apnea": ["sleep apnea", "obstructive sleep apnea", "CPAP therapy", "sleep disordered breathing"],
    "pulmonary_hypertension": ["pulmonary hypertension", "pulmonary arterial hypertension"],
    "lung_transplant": ["lung transplant", "lung transplantation", "bronchiolitis obliterans"],
    # 消化系统 (10)
    "ibd": ["inflammatory bowel disease", "IBD treatment", "biologics IBD"],
    "crohn": ["crohn disease", "crohn treatment", "fistulizing crohn"],
    "ulcerative_colitis": ["ulcerative colitis", "UC treatment", "colectomy"],
    "liver_disease": ["liver disease hepatitis", "chronic liver disease", "liver cirrhosis"],
    "cirrhosis": ["cirrhosis", "hepatic encephalopathy", "portal hypertension", "ascites"],
    "celiac": ["celiac disease", "gluten sensitivity", "gluten free diet"],
    "gerd": ["gastroesophageal reflux", "GERD treatment", "proton pump inhibitor", "barrett esophagus"],
    "pancreatitis": ["pancreatitis", "acute pancreatitis", "chronic pancreatitis"],
    "gi_bleeding": ["gastrointestinal bleeding", "upper GI bleeding", "lower GI bleeding"],
    "liver_transplant": ["liver transplant", "liver transplantation", "hepatic failure"],
    # 肾脏与泌尿 (8)
    "ckd": ["chronic kidney disease", "CKD stages", "renal insufficiency", "dialysis"],
    "aki": ["acute kidney injury", "acute renal failure", "AKI treatment"],
    "kidney_transplant": ["kidney transplant", "renal transplant", "transplant rejection"],
    "uti": ["urinary tract infection", "pyelonephritis", "cystitis", "recurrent UTI"],
    "nephrotic": ["nephrotic syndrome", "proteinuria", "glomerulonephritis"],
    "pkd": ["polycystic kidney", "ADPKD", "kidney cyst"],
    "dialysis": ["hemodialysis", "peritoneal dialysis", "dialysis access"],
    "electrolyte": ["electrolyte imbalance", "hyponatremia", "hyperkalemia", "acid base disorder"],
    # 免疫与风湿 (10)
    "rheumatoid": ["rheumatoid arthritis", "RA treatment", "methotrexate", "biologic DMARD"],
    "lupus": ["systemic lupus erythematosus", "SLE treatment", "lupus nephritis"],
    "psoriasis": ["psoriasis treatment", "psoriatic arthritis", "biologic psoriasis"],
    "autoimmune": ["autoimmune diseases", "autoimmunity", "immunosuppression"],
    "spondylitis": ["ankylosing spondylitis", "axial spondyloarthritis", "TNF inhibitor"],
    "sjogren": ["sjogren syndrome", "sicca syndrome", "dry eye dry mouth"],
    "vasculitis": ["vasculitis", "giant cell arteritis", "ANCA vasculitis"],
    "scleroderma": ["scleroderma", "systemic sclerosis", "raynaud phenomenon"],
    "myositis": ["inflammatory myopathy", "dermatomyositis", "polymyositis"],
    "allergy": ["allergic disease", "anaphylaxis", "food allergy", "drug allergy"],
    # 感染性疾病 (10)
    "hiv": ["HIV AIDS treatment", "antiretroviral therapy", "HIV prevention", "PrEP"],
    "covid": ["covid-19", "SARS-CoV-2", "coronavirus treatment", "COVID vaccine"],
    "sepsis": ["sepsis infection", "septic shock", "sepsis management"],
    "malaria": ["malaria treatment", "plasmodium", "antimalarial drug"],
    "hepatitis_b": ["hepatitis B treatment", "HBV", "hepatitis B vaccine"],
    "hepatitis_c": ["hepatitis C treatment", "HCV", "direct acting antiviral"],
    "influenza": ["influenza", "flu treatment", "influenza vaccine", "oseltamivir"],
    "meningitis": ["bacterial meningitis", "viral meningitis", "meningococcal"],
    "fungal": ["fungal infection", "candidiasis", "aspergillosis", "antifungal therapy"],
    "antibiotic": ["antibiotic resistance", "antimicrobial stewardship", "MRSA", "multidrug resistant"],
    # 精神与心理 (8)
    "mental_health": ["mental health disorders", "psychiatric disorder", "mental illness"],
    "depression": ["anxiety depression", "major depressive disorder", "antidepressant"],
    "schizophrenia": ["schizophrenia treatment", "psychosis", "antipsychotic"],
    "bipolar": ["bipolar disorder", "mood stabilizer", "lithium treatment"],
    "ptsd": ["post traumatic stress", "PTSD treatment", "trauma therapy"],
    "eating": ["eating disorders", "anorexia nervosa", "bulimia nervosa"],
    "addiction": ["substance abuse", "opioid addiction", "alcohol use disorder"],
    "adhd": ["attention deficit hyperactivity", "ADHD treatment", "stimulant medication"],
    # 其他 (7)
    "pain": ["chronic pain management", "pain treatment", "opioid therapy", "multimodal analgesia"],
    "sleep": ["sleep disorders insomnia", "insomnia treatment", "sleep hygiene"],
    "pediatric": ["pediatric diseases", "childhood illness", "neonatal", "pediatric care"],
    "geriatric": ["geriatric medicine", "elderly care", "frailty", "polypharmacy"],
    "palliative": ["palliative care", "hospice", "end of life care"],
    "rehabilitation": ["physical rehabilitation", "occupational therapy", "physical therapy"],
    "emergency": ["emergency medicine", "critical care", "ICU", "resuscitation"],
}

MEDICAL_TOPICS = list(MEDICAL_TOPICS_EXPANDED.keys())


class AsyncPubMedCrawler:
    """高性能异步PubMed爬虫"""
    
    def __init__(self, email: str = None, api_key: str = None, 
                 max_concurrent: int = 10, checkpoint_dir: Path = None):
        self.email = email or PUBMED_EMAIL
        self.api_key = api_key or PUBMED_API_KEY
        self.max_concurrent = max_concurrent if self.api_key else 3
        self.target_size_bytes = TARGET_SIZE_BYTES
        
        # 速率控制 (有API Key: 10/s, 无: 3/s)
        self.rate_limit = 10 if self.api_key else 3
        self.request_interval = 1.0 / self.rate_limit
        self.last_request_time = 0
        self.rate_lock = asyncio.Lock()
        
        # 断点续传
        self.checkpoint_dir = checkpoint_dir or (RAW_DATA_DIR / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.lock = threading.Lock()
        self.all_pmids: Set[str] = set()
        self.all_articles: List[Dict] = []
        self.completed_topics: Set[str] = set()
        self.topic_sizes: Dict[str, int] = {}
        self.partial_topics: Dict[str, List[Dict]] = {}
        
        # 统计
        self.stats = {"total": 0, "success": 0, "failed": 0, "retries": 0}
        
        # 加载checkpoint
        self._load_checkpoint()
    
    async def _rate_limit_wait(self):
        """速率限制等待"""
        async with self.rate_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.request_interval:
                await asyncio.sleep(self.request_interval - elapsed)
            self.last_request_time = time.time()
    
    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str, 
                                 params: dict, max_retries: int = 5) -> Optional[str]:
        """带重试的异步请求"""
        for attempt in range(max_retries):
            try:
                await self._rate_limit_wait()
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status == 429:  # Rate limited
                        delay = 2 ** attempt + random.uniform(0, 1)
                        logger.warning(f"Rate limited, waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(f"HTTP {resp.status}")
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt + random.uniform(0, 1)
                    self.stats["retries"] += 1
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed: {e}")
        return None

    async def search_articles(self, session: aiohttp.ClientSession, 
                              query: str, max_results: int = 9999) -> List[str]:
        """异步搜索文章ID"""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 9999),
            "sort": "relevance",
            "retmode": "json",
            "email": self.email
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        text = await self._fetch_with_retry(session, PUBMED_SEARCH_URL, params)
        if not text:
            return []
        
        try:
            data = json.loads(text)
            ids = data.get("esearchresult", {}).get("idlist", [])
            count = data.get("esearchresult", {}).get("count", "0")
            logger.info(f"[{query[:30]}] 总数: {count}, 获取: {len(ids)}")
            return ids
        except Exception as e:
            logger.error(f"Parse search result failed: {e}")
            return []
    
    async def fetch_articles_batch(self, session: aiohttp.ClientSession,
                                   id_list: List[str], topic: str) -> List[Dict]:
        """异步批量获取文章详情"""
        if not id_list:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "rettype": "xml",
            "retmode": "xml",
            "email": self.email
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        text = await self._fetch_with_retry(session, PUBMED_FETCH_URL, params)
        if not text:
            return []
        
        return self._parse_xml_articles(text, topic)
    
    def _parse_xml_articles(self, xml_text: str, topic: str) -> List[Dict]:
        """解析XML格式的文章数据 (使用lxml更快)"""
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for article_elem in root.findall('.//PubmedArticle'):
                article = self._parse_single_article(article_elem, topic)
                if article:
                    articles.append(article)
        except Exception as e:
            logger.warning(f"XML parse error: {e}")
        return articles
    
    def _parse_single_article(self, elem, topic: str) -> Optional[Dict]:
        """解析单篇文章"""
        try:
            # PMID
            pmid_elem = elem.find('.//PMID')
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text
            
            # Title
            title_elem = elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_parts = []
            for abs_elem in elem.findall('.//AbstractText'):
                if abs_elem.text:
                    abstract_parts.append(abs_elem.text)
            abstract = " ".join(abstract_parts)
            
            if not abstract or len(abstract) < 100:
                return None
            
            # Authors (前5个)
            authors = []
            for author in elem.findall('.//Author')[:5]:
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None:
                    name = f"{forename.text if forename is not None else ''} {lastname.text}".strip()
                    if name:
                        authors.append(name)
            
            # Publication date
            pub_date = ""
            year_elem = elem.find('.//PubDate/Year')
            month_elem = elem.find('.//PubDate/Month')
            if year_elem is not None:
                pub_date = year_elem.text
                if month_elem is not None:
                    pub_date += f"-{month_elem.text}"
            
            # MeSH terms (前10个)
            mesh_terms = []
            for mesh in elem.findall('.//MeshHeading/DescriptorName')[:10]:
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            # Keywords (前10个)
            keywords = []
            for kw in elem.findall('.//Keyword')[:10]:
                if kw.text:
                    keywords.append(kw.text)
            
            return {
                'pmid': pmid,
                'title': title or "",
                'abstract': abstract,
                'authors': authors,
                'pub_date': pub_date,
                'keywords': keywords,
                'mesh_terms': mesh_terms,
                'topic': topic,
                'full_text': f"{title}\n\n{abstract}"
            }
        except Exception:
            return None
    
    def _calculate_size(self, articles: List[Dict]) -> int:
        """计算JSON大小"""
        return len(json.dumps(articles, ensure_ascii=False).encode('utf-8'))
    
    # ==================== 断点续传 ====================
    
    def _get_checkpoint_path(self, topic: str) -> Path:
        safe_name = topic.replace(" ", "_").replace("/", "_")
        return self.checkpoint_dir / f"checkpoint_{safe_name}.json"
    
    def _save_topic_checkpoint(self, topic: str, articles: List[Dict], completed: bool = False):
        """保存主题checkpoint"""
        size_bytes = self._calculate_size(articles)
        checkpoint = {
            "topic": topic,
            "completed": completed,
            "article_count": len(articles),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "pmids": [a["pmid"] for a in articles],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        path = self._get_checkpoint_path(topic)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        data_path = path.with_suffix('.data.json')
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False)
        
        self.topic_sizes[topic] = size_bytes
    
    def _load_topic_checkpoint(self, topic: str) -> Optional[List[Dict]]:
        """加载主题checkpoint"""
        path = self._get_checkpoint_path(topic)
        data_path = path.with_suffix('.data.json')
        
        if not path.exists() or not data_path.exists():
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            size_mb = checkpoint.get("size_mb", 0)
            
            with open(data_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            self.topic_sizes[topic] = int(size_mb * 1024 * 1024)
            
            if checkpoint.get("completed") and size_mb >= TARGET_SIZE_MB:
                logger.info(f"📂 已完成: {topic} ({len(articles)} 篇, {size_mb:.1f}MB)")
                return articles
            else:
                logger.info(f"📂 未完成: {topic} ({len(articles)} 篇, {size_mb:.1f}MB)")
                self.partial_topics[topic] = articles
                return None
        except Exception as e:
            logger.warning(f"Load checkpoint failed [{topic}]: {e}")
        return None
    
    def _load_checkpoint(self):
        """加载所有checkpoint"""
        logger.info("🔍 检查断点续传...")
        
        for topic in MEDICAL_TOPICS:
            articles = self._load_topic_checkpoint(topic)
            if articles:
                self.completed_topics.add(topic)
                for article in articles:
                    if article["pmid"] not in self.all_pmids:
                        self.all_pmids.add(article["pmid"])
                        self.all_articles.append(article)
        
        if self.completed_topics:
            total_size = sum(self.topic_sizes.values()) / (1024 * 1024)
            logger.info(f"✅ 已恢复 {len(self.completed_topics)} 个主题, {len(self.all_articles):,} 篇, {total_size:.1f}MB")
    
    def _save_global_checkpoint(self):
        """保存全局进度"""
        global_path = self.checkpoint_dir / "global_progress.json"
        total_size = sum(self.topic_sizes.values())
        progress = {
            "completed_topics": list(self.completed_topics),
            "total_articles": len(self.all_articles),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "topic_sizes": {k: round(v / (1024 * 1024), 2) for k, v in self.topic_sizes.items()},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(global_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    # ==================== 核心爬取逻辑 ====================
    
    async def crawl_topic(self, session: aiohttp.ClientSession, topic: str) -> List[Dict]:
        """异步爬取单个主题"""
        if topic in self.completed_topics:
            logger.info(f"⏭️ 跳过: {topic}")
            return []
        
        sub_queries = MEDICAL_TOPICS_EXPANDED.get(topic, [topic])
        
        # 加载已有数据
        articles = []
        used_pmids = set()
        if topic in self.partial_topics:
            articles = self.partial_topics[topic].copy()
            used_pmids = {a['pmid'] for a in articles}
            current_size = self._calculate_size(articles)
            logger.info(f"🔄 增量: {topic} (已有: {len(articles)} 篇, {current_size/(1024*1024):.1f}MB)")
        else:
            current_size = 0
            logger.info(f"🔍 开始: {topic} (目标: {TARGET_SIZE_MB}MB, 子查询: {len(sub_queries)}个)")
        
        batch_size = 200  # 每批获取数量
        
        for sub_query in sub_queries:
            if current_size >= self.target_size_bytes:
                break
            
            logger.info(f"   [{topic}] 查询: {sub_query}")
            id_list = await self.search_articles(session, sub_query, max_results=9999)
            
            if not id_list:
                continue
            
            new_ids = [pid for pid in id_list if pid not in used_pmids]
            logger.info(f"   [{topic}] 新增ID: {len(new_ids)}")
            
            # 并发获取文章详情
            for i in range(0, len(new_ids), batch_size):
                if current_size >= self.target_size_bytes:
                    break
                
                batch_ids = new_ids[i:i + batch_size]
                batch_articles = await self.fetch_articles_batch(session, batch_ids, topic)
                
                for article in batch_articles:
                    if article['pmid'] not in used_pmids:
                        used_pmids.add(article['pmid'])
                        articles.append(article)
                
                current_size = self._calculate_size(articles)
                current_mb = current_size / (1024 * 1024)
                
                # 每5批报告进度
                if (i // batch_size + 1) % 5 == 0:
                    logger.info(f"   [{topic}] 进度: {len(articles)} 篇, {current_mb:.1f}MB")
                    self._save_topic_checkpoint(topic, articles, completed=False)
        
        # 保存结果
        with self.lock:
            new_count = 0
            for article in articles:
                if article['pmid'] not in self.all_pmids:
                    self.all_pmids.add(article['pmid'])
                    self.all_articles.append(article)
                    new_count += 1
            self.stats["success"] += new_count
            self.completed_topics.add(topic)
        
        final_size = self._calculate_size(articles) / (1024 * 1024)
        completed = final_size >= TARGET_SIZE_MB
        self._save_topic_checkpoint(topic, articles, completed=completed)
        self._save_global_checkpoint()
        
        status = "✅" if completed else "⚠️"
        logger.info(f"{status} [{topic}] 完成: {len(articles)} 篇, {final_size:.1f}MB")
        return articles
    
    async def crawl_topics_concurrent(self, topics: List[str], max_concurrent: int = 3):
        """并发爬取多个主题"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_with_semaphore(session, topic):
            async with semaphore:
                return await self.crawl_topic(session, topic)
        
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=120, connect=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [crawl_with_semaphore(session, topic) for topic in topics]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for topic, result in zip(topics, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ [{topic}] 失败: {result}")
                    self.stats["failed"] += 1
        
        return results
    
    def crawl_all_topics(self, topics: List[str] = None, max_concurrent: int = 3):
        """爬取所有主题 (同步入口)"""
        topics = topics or MEDICAL_TOPICS
        pending_topics = [t for t in topics if t not in self.completed_topics]
        
        total_size = sum(self.topic_sizes.values()) / (1024 * 1024)
        
        logger.info("=" * 60)
        logger.info(f"🚀 异步PubMed爬虫启动")
        logger.info(f"   总主题: {len(topics)}")
        logger.info(f"   待爬取: {len(pending_topics)}")
        logger.info(f"   已完成: {len(self.completed_topics)}")
        logger.info(f"   当前大小: {total_size:.1f}MB")
        logger.info(f"   目标大小: {len(topics) * TARGET_SIZE_MB}MB ({len(topics) * TARGET_SIZE_MB / 1024:.1f}GB)")
        logger.info(f"   并发数: {max_concurrent} 主题, {self.max_concurrent} 请求")
        logger.info(f"   速率限制: {self.rate_limit} 请求/秒")
        logger.info("=" * 60)
        
        if not pending_topics:
            logger.info("✅ 所有主题已完成!")
            return self.all_articles
        
        start_time = time.time()
        
        # 运行异步爬取
        asyncio.run(self.crawl_topics_concurrent(pending_topics, max_concurrent))
        
        elapsed = time.time() - start_time
        
        # 保存最终结果
        output_file = RAW_DATA_DIR / "pubmed_articles_all.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_articles, f, ensure_ascii=False)
        
        self._print_statistics(elapsed, output_file)
        return self.all_articles
    
    def _print_statistics(self, elapsed: float, output_file: Path):
        """打印统计"""
        total_size = output_file.stat().st_size / (1024 * 1024)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 爬取完成")
        logger.info("=" * 60)
        logger.info(f"   总文章: {len(self.all_articles):,}")
        logger.info(f"   总大小: {total_size:.1f}MB ({total_size/1024:.2f}GB)")
        logger.info(f"   耗时: {elapsed/60:.1f} 分钟")
        logger.info(f"   速度: {len(self.all_articles)/max(elapsed,1):.1f} 篇/秒")
        logger.info(f"   重试: {self.stats['retries']}")
        logger.info(f"   保存: {output_file}")
    
    def clear_checkpoints(self):
        """清除checkpoint"""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.all_pmids.clear()
        self.all_articles.clear()
        self.completed_topics.clear()
        self.topic_sizes.clear()
        logger.info("🗑️ Checkpoint已清除")


# 兼容旧接口
PubMedCrawler = AsyncPubMedCrawler


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="PubMed高性能异步爬虫")
    parser.add_argument("--workers", type=int, default=3, help="并发主题数")
    parser.add_argument("--concurrent", type=int, default=10, help="并发请求数")
    parser.add_argument("--clear", action="store_true", help="清除checkpoint")
    args = parser.parse_args()
    
    crawler = AsyncPubMedCrawler(max_concurrent=args.concurrent)
    
    if args.clear:
        crawler.clear_checkpoints()
    
    crawler.crawl_all_topics(max_concurrent=args.workers)


if __name__ == "__main__":
    main()
