import os
import json
import time
import uuid
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context

from rag_engine import RAGEngine
from llm_client import DeepSeekClient

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = BASE_DIR / "config.json"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".docx"}

rag = RAGEngine(str(DATA_DIR), str(INDEX_DIR))
# 预加载嵌入模型（避免首次查询等待）
_ = rag.encoder

# ── Session 管理 ───────────────────────────────────────

SESSION_TIMEOUT = 1800  # 30 分钟
MAX_HISTORY_TURNS = 6   # 保留最近 3 轮对话（6 条消息）
_sessions = {}
_sessions_lock = threading.Lock()


def _get_session(session_id):
    """获取或创建会话。"""
    now = time.time()
    with _sessions_lock:
        # 清理过期会话
        expired = [sid for sid, s in _sessions.items()
                   if now - s["last_active"] > SESSION_TIMEOUT]
        for sid in expired:
            del _sessions[sid]

        if session_id not in _sessions:
            _sessions[session_id] = {
                "messages": [],
                "last_active": now,
            }
        else:
            _sessions[session_id]["last_active"] = now
        return _sessions[session_id]


def _add_to_history(session_id, role, content):
    """添加消息到会话历史。"""
    session = _get_session(session_id)
    session["messages"].append({"role": role, "content": content})
    # 只保留最近 N 条
    if len(session["messages"]) > MAX_HISTORY_TURNS:
        session["messages"] = session["messages"][-MAX_HISTORY_TURNS:]


# ── API Key 持久化 ─────────────────────────────────────

def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(config):
    try:
        CONFIG_FILE.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[Config] 保存失败: {e}")


_config = load_config()
# 优先环境变量，其次配置文件
api_key = os.environ.get("DEEPSEEK_API_KEY") or _config.get("api_key")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/session", methods=["GET"])
def api_session():
    """创建新会话。"""
    session_id = str(uuid.uuid4())
    _get_session(session_id)
    return jsonify({"session_id": session_id})


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
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
    global api_key
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    top_k = data.get("top_k", 5)
    session_id = data.get("session_id", "")

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 获取会话历史
    history = []
    if session_id:
        session = _get_session(session_id)
        history = list(session["messages"])

    # LLM 模式下取消阈值限制
    threshold = 0.0 if api_key else 0.05
    results = rag.query(question, top_k=top_k, threshold=threshold)

    if not results:
        _add_to_history(session_id, "user", question)
        _add_to_history(session_id, "assistant",
                        "抱歉，我在知识库中没有找到与您问题相关的内容。")
        return jsonify({
            "answer": "抱歉，我在知识库中没有找到与您问题相关的内容。请尝试上传更多商品文档到 data 目录中。",
            "sources": [],
            "found": False,
        })

    sources = [{"source": r["source"], "score": r["score"]} for r in results]

    # ── 纯检索模式 ──
    if not api_key:
        answer = "\n\n".join(r["text"] for r in results)
        _add_to_history(session_id, "user", question)
        _add_to_history(session_id, "assistant", answer)
        return jsonify({
            "answer": answer,
            "sources": sources,
            "details": results,
            "found": True,
            "llm_enabled": False,
        })

    # ── LLM 流式模式 ──
    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        client = DeepSeekClient(api_key)
        full_response = []
        has_output = False
        llm_error = None

        try:
            for chunk in client.chat_stream(question, results, history=history):
                if chunk.startswith("__ERROR__:"):
                    llm_error = chunk[len("__ERROR__:"):]
                    break
                has_output = True
                full_response.append(chunk)
                yield f"data: {json.dumps({'type': 'delta', 'data': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            llm_error = str(e)

        # LLM 失败时自动回退
        if llm_error:
            if has_output:
                note = f"\n\n⚠️ AI 回答中断（{llm_error}），以下为知识库补充检索结果：\n\n"
            else:
                note = f"⚠️ AI 回答暂不可用（{llm_error}），以下为知识库检索结果：\n\n"
            note += "\n\n".join(r["text"] for r in results)
            full_response.append(note)
            yield f"data: {json.dumps({'type': 'delta', 'data': note}, ensure_ascii=False)}\n\n"

        # 保存对话历史
        _add_to_history(session_id, "user", question)
        _add_to_history(session_id, "assistant", "".join(full_response))

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
    return jsonify({"documents": rag.list_documents()})


@app.route("/api/delete", methods=["POST"])
def api_delete():
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
    return jsonify(rag.get_stats())


@app.route("/api/download/<filename>", methods=["GET"])
def api_download(filename):
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "非法文件名"}), 400
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_from_directory(str(DATA_DIR), filename, as_attachment=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "stats": rag.get_stats()})


if __name__ == "__main__":
    print("=" * 50)
    print("  电商 RAG 知识库系统")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  索引目录: {INDEX_DIR}")
    print(f"  引擎状态: {rag.get_stats()}")
    print(f"  LLM 模型: deepseek-v4-pro")
    print(f"  LLM 状态: {'已启用' if api_key else '未启用（纯检索模式）'}")
    print("  访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
