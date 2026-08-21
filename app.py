import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context

from rag_engine import RAGEngine
from llm_client import DeepSeekClient

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = Path(__file__).parent / "config.json"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".docx"}

rag = RAGEngine(str(DATA_DIR))


# ── API Key 持久化 ─────────────────────────────────────

def load_config():
    """从配置文件加载设置。"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(config):
    """保存设置到配置文件。"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)
    except Exception as e:
        print(f"[Config] 保存失败: {e}")


# 启动时从配置文件加载 API Key
_config = load_config()
api_key = _config.get("api_key")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """管理 DeepSeek API Key，持久化到 config.json。"""
    global api_key
    if request.method == "GET":
        return jsonify({
            "llm_enabled": api_key is not None,
            "model": "deepseek-v4-pro",
        })

    data = request.get_json(silent=True) or {}
    key = (data.get("api_key") or "").strip()
    if key:
        api_key = key
        save_config({"api_key": key})
        return jsonify({
            "success": True,
            "message": "API 密钥已设置，AI 回答已启用",
            "llm_enabled": True,
        })
    else:
        api_key = None
        save_config({})
        return jsonify({
            "success": True,
            "message": "API 密钥已清除，已切换为纯检索模式",
            "llm_enabled": False,
        })


@app.route("/api/query", methods=["POST"])
def api_query():
    """接受用户问题。有 API Key 时走 SSE 流式 LLM 回答，否则返回纯检索结果。"""
    global api_key
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    top_k = data.get("top_k", 5)

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # LLM 模式下取消阈值限制，让 LLM 判断相关性
    # 纯检索模式保留阈值过滤低质量匹配
    threshold = 0.0 if api_key else 0.05
    results = rag.query(question, top_k=top_k, threshold=threshold)

    if not results:
        return jsonify({
            "answer": "抱歉，我在知识库中没有找到与您问题相关的内容。请尝试上传更多商品文档到 data 目录中。",
            "sources": [],
            "found": False,
        })

    sources = [{"source": r["source"], "score": r["score"]} for r in results]

    # ── 纯检索模式（无 API Key）──
    if not api_key:
        answer = "\n\n".join(r["text"] for r in results)
        return jsonify({
            "answer": answer,
            "sources": sources,
            "details": results,
            "found": True,
            "llm_enabled": False,
        })

    # ── LLM 流式模式（含回退）──
    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        client = DeepSeekClient(api_key)
        has_output = False
        llm_error = None

        try:
            for chunk in client.chat_stream(question, results):
                if chunk.startswith("__ERROR__:"):
                    llm_error = chunk[len("__ERROR__:"):]
                    break
                has_output = True
                yield f"data: {json.dumps({'type': 'delta', 'data': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            llm_error = str(e)

        # LLM 失败时自动回退到检索结果
        if llm_error:
            if has_output:
                note = f"\n\n⚠️ AI 回答中断（{llm_error}），以下为知识库补充检索结果：\n\n"
            else:
                note = f"⚠️ AI 回答暂不可用（{llm_error}），以下为知识库检索结果：\n\n"
            note += "\n\n".join(r["text"] for r in results)
            yield f"data: {json.dumps({'type': 'delta', 'data': note}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """上传文档到 data 目录。"""
    if "file" not in request.files:
        return jsonify({"error": "未检测到上传文件"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": f"不支持的文件类型 {ext}，仅支持 {', '.join(SUPPORTED_EXTENSIONS)}"
        }), 400

    content = file.read()
    name = rag.add_document(file.filename, content)

    return jsonify({
        "success": True,
        "message": f"文件 {name} 上传成功，索引已更新",
        "stats": rag.get_stats(),
    })


@app.route("/api/documents", methods=["GET"])
def api_documents():
    """列出 data 目录中的所有文档。"""
    return jsonify({"documents": rag.list_documents()})


@app.route("/api/delete", methods=["POST"])
def api_delete():
    """删除指定文档。"""
    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "").strip()

    if not filename:
        return jsonify({"error": "文件名不能为空"}), 400

    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "非法文件名"}), 400

    if rag.delete_document(filename):
        return jsonify({
            "success": True,
            "message": f"文件 {filename} 已删除，索引已更新",
            "stats": rag.get_stats(),
        })
    else:
        return jsonify({"error": f"文件 {filename} 不存在"}), 404


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """返回引擎统计信息。"""
    return jsonify(rag.get_stats())


@app.route("/api/download/<filename>", methods=["GET"])
def api_download(filename):
    """下载指定文档。"""
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "非法文件名"}), 400
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_from_directory(str(DATA_DIR), filename, as_attachment=True)


if __name__ == "__main__":
    print("=" * 50)
    print("  电商 RAG 知识库系统")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  已加载文档: {rag.get_stats()}")
    print(f"  LLM 模型: deepseek-v4-pro")
    print(f"  LLM 状态: {'已启用' if api_key else '未启用（纯检索模式）'}")
    print("  访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
