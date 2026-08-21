import os
import json
import csv
import re
from pathlib import Path
from docx import Document

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 电商领域自定义词典
CUSTOM_WORDS = [
    "尺码", "型号", "面料", "材质", "商品编号", "库存",
    "包邮", "退换货", "售后服务", "发货时间", "产地",
    "男装", "女装", "童装", "鞋类", "配饰", "电子产品",
    "家居用品", "食品", "美妆", "运动户外",
]

for w in CUSTOM_WORDS:
    jieba.add_word(w)


def tokenize(text):
    """jieba 分词，返回空格分隔的词串。"""
    words = jieba.cut(text)
    return " ".join(w.strip() for w in words if w.strip())


class RAGEngine:
    """基于 TF-IDF + 余弦相似度的轻量 RAG 检索引擎。"""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = []          # [{"text":..., "source":..., "meta":...}]
        self.vectorizer = None
        self.tfidf_matrix = None
        self._rebuild()

    # ── 文档加载 ──────────────────────────────────────────

    def _load_file(self, filepath):
        """读取单个文件，返回 (text, meta) 列表。"""
        ext = filepath.suffix.lower()
        results = []

        if ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            results.append((text, {"source": filepath.name, "type": "text"}))

        elif ext == ".csv":
            with open(filepath, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    text = "；".join(f"{k}：{v}" for k, v in row.items() if v)
                    results.append((text, {
                        "source": filepath.name, "type": "csv", "row": i + 2
                    }))

        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    text = self._json_to_text(item)
                    results.append((text, {
                        "source": filepath.name, "type": "json", "index": i
                    }))
            elif isinstance(data, dict):
                text = self._json_to_text(data)
                results.append((text, {
                    "source": filepath.name, "type": "json"
                }))

        elif ext == ".docx":
            doc = Document(str(filepath))
            parts = []
            for para in doc.paragraphs:
                t = para.text.strip()
                if t:
                    parts.append(t)
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append("；".join(cells))
            text = "\n\n".join(parts)
            results.append((text, {"source": filepath.name, "type": "docx"}))

        return results

    @staticmethod
    def _json_to_text(obj, prefix=""):
        """将 JSON 对象转为可读文本。"""
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{k}：{RAGEngine._json_to_text(v)}")
                else:
                    lines.append(f"{k}：{v}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    lines.append(f"[{i}] {RAGEngine._json_to_text(item)}")
                else:
                    lines.append(str(item))
        else:
            lines.append(str(obj))
        return "；".join(lines)

    def _split_chunks(self, text, source, meta, max_len=300):
        """将长文本按段落切分为 chunk。"""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 长段落再切分
            while len(para) > max_len:
                cut = para[:max_len]
                # 尽量在句号/分号处断
                for sep in ['。', '；', '!', '?', '；', '\n']:
                    pos = cut.rfind(sep)
                    if pos > max_len // 2:
                        cut = para[:pos + 1]
                        break
                chunks.append(cut)
                para = para[len(cut):]
            if para:
                chunk_meta = dict(meta)
                chunk_meta["source"] = source
                chunk_meta["preview"] = para[:80]
                chunks.append({
                    "text": para,
                    "source": source,
                    "meta": chunk_meta,
                })
        return chunks

    # ── 索引构建 ──────────────────────────────────────────

    def _rebuild(self):
        """重新加载所有文件并构建 TF-IDF 索引。"""
        self.chunks = []
        supported = {".txt", ".md", ".csv", ".json", ".docx"}

        for filepath in sorted(self.data_dir.iterdir()):
            if filepath.is_file() and filepath.suffix.lower() in supported:
                try:
                    docs = self._load_file(filepath)
                    for text, meta in docs:
                        new_chunks = self._split_chunks(
                            text, filepath.name, meta
                        )
                        self.chunks.extend(new_chunks)
                except Exception as e:
                    print(f"[RAG] 加载 {filepath.name} 失败: {e}")

        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            print("[RAG] 数据目录无可用文档")
            return

        corpus = [tokenize(c["text"]) for c in self.chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"[RAG] 索引构建完成，共 {len(self.chunks)} 个文本块")

    # ── 检索 ──────────────────────────────────────────────

    @staticmethod
    def _expand_query(query):
        """扩展口语化查询，补充正式术语以提升 TF-IDF 命中率。"""
        expansions = []

        # 中文身高 → 厘米数字（一米八 → 180）
        height_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
                      "六": "6", "七": "7", "八": "8", "九": "9"}
        for m in re.finditer(r"一米(.)", query):
            digit = height_map.get(m.group(1))
            if digit:
                expansions.append(f"1{digit}0")

        # 口语词 → 文档中的正式术语
        term_map = {
            "男士": ["男装", "男"],
            "女士": ["女装", "女"],
            "多大码": ["尺码", "尺寸", "适合"],
            "穿什么": ["适合", "尺码"],
            "穿多大": ["适合", "尺码"],
            "多少钱": ["价格", "售价"],
            "包邮吗": ["运费", "包邮"],
            "能退吗": ["退换货", "退货"],
            "多久到": ["时效", "发货时间"],
            "有什么商品": ["商品", "产品"],
            "有哪些商品": ["商品", "产品"],
            "蓝牙耳机": ["耳机", "SC-505", "声科达"],
            "T恤": ["T 恤", "短袖"],
            "裤子": ["男裤", "女裤", "休闲裤", "牛仔裤"],
            "裙子": ["连衣裙"],
            "上衣": ["T 恤", "卫衣", "衬衫"],
            "冲锋衣": ["童装", "防风"],
        }
        for colloquial, formal_list in term_map.items():
            if colloquial in query:
                expansions.extend(formal_list)

        if expansions:
            return query + " " + " ".join(set(expansions))
        return query

    def query(self, question, top_k=5, threshold=0.05):
        """检索与问题最相关的文本块。"""
        if self.vectorizer is None or not self.chunks:
            return []

        expanded = self._expand_query(question)
        q_vec = self.vectorizer.transform([tokenize(expanded)])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # 排序取 top_k
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked[:top_k]:
            score = float(scores[idx])
            if score < threshold:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": round(score, 4),
                "meta": chunk.get("meta", {}),
            })
        return results

    # ── 文档管理 ──────────────────────────────────────────

    def list_documents(self):
        """列出 data 目录中的所有文档。"""
        docs = []
        supported = {".txt", ".md", ".csv", ".json", ".docx"}
        for filepath in sorted(self.data_dir.iterdir()):
            if filepath.is_file() and filepath.suffix.lower() in supported:
                stat = filepath.stat()
                docs.append({
                    "name": filepath.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
        return docs

    def add_document(self, filename, content_bytes):
        """保存上传的文档并重建索引。"""
        filepath = self.data_dir / filename
        with open(filepath, "wb") as f:
            f.write(content_bytes)
        self._rebuild()
        return filepath.name

    def delete_document(self, filename):
        """删除文档并重建索引。"""
        filepath = self.data_dir / filename
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            self._rebuild()
            return True
        return False

    def get_stats(self):
        """返回引擎统计信息。"""
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(self.list_documents()),
        }
