import json
import requests


class DeepSeekClient:
    """DeepSeek API 客户端，支持流式输出 + 多轮对话。"""

    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-v4-pro"

    SYSTEM_PROMPT = (
        "你是一个电商智能客服助手。请根据以下知识库检索结果回答用户的问题。\n\n"
        "规则：\n"
        "1. 优先依据下方知识库内容进行回答，不可编造不存在的商品信息\n"
        "2. 理解用户的口语化表达并映射到知识库内容\n"
        "   （例如：男士=男装、女士=女装、一米八=180cm、多大码=适合什么尺码等）\n"
        "3. 若知识库中有相关信息，请结合知识库数据给出具体建议\n"
        "4. 若知识库中确实没有相关信息，请说明「知识库中暂无相关内容」\n"
        "5. 回答要简洁、准确、条理清晰\n"
        "6. 涉及尺码、价格等具体数据时，请清晰列出\n"
        "7. 使用中文回答\n"
        "8. 这是多轮对话，用户的问题可能引用上文（如「这个多少钱」「那个有什么颜色」），"
        "请结合对话历史理解用户意图\n\n"
        "知识库检索结果：\n{context}"
    )

    ERROR_TRANSLATIONS = {
        "insufficient balance": "DeepSeek 账户余额不足，请前往 platform.deepseek.com 充值",
        "invalid api key": "API 密钥无效，请检查密钥是否正确",
        "rate limit": "API 调用频率超限，请稍后重试",
        "model not found": "模型名称错误，请检查模型配置",
    }

    def __init__(self, api_key):
        self.api_key = api_key

    def _translate_error(self, error_msg):
        lower = error_msg.lower()
        for en, zh in self.ERROR_TRANSLATIONS.items():
            if en in lower:
                return zh
        return error_msg

    def _build_messages(self, question, context_chunks, history=None):
        """构建多轮对话消息列表。"""
        context = "\n\n---\n\n".join(
            f"[来源: {c['source']}] {c['text']}" for c in context_chunks
        )
        system_content = self.SYSTEM_PROMPT.format(context=context)

        messages = [{"role": "system", "content": system_content}]

        # 添加对话历史
        if history:
            for msg in history:
                content = msg.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                messages.append({
                    "role": msg["role"],
                    "content": content,
                })

        # 当前问题
        messages.append({"role": "user", "content": question})
        return messages

    def chat_stream(self, question, context_chunks, history=None):
        """流式调用，逐块 yield 文本。支持多轮对话。"""
        messages = self._build_messages(question, context_chunks, history)

        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.MODEL,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 1024,
                },
                stream=True,
                timeout=60,
            )
        except requests.exceptions.ConnectionError:
            yield "__ERROR__: 无法连接到 DeepSeek API，请检查网络"
            return
        except requests.exceptions.Timeout:
            yield "__ERROR__: API 请求超时，请稍后重试"
            return

        if response.status_code == 401:
            yield "__ERROR__: API 密钥无效，请检查密钥是否正确"
            return
        if response.status_code == 429:
            yield "__ERROR__: API 调用频率超限，请稍后重试"
            return
        if response.status_code != 200:
            try:
                err = response.json().get("error", {}).get("message", "")
            except Exception:
                err = f"HTTP {response.status_code}"
            yield f"__ERROR__: {self._translate_error(err)}"
            return

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
