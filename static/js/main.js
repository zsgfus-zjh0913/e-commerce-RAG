// ═══ DOM 引用 ═══
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const docCount = document.getElementById("doc-count");
const statDocs = document.getElementById("stat-docs");
const statChunks = document.getElementById("stat-chunks");
const apiKeyInput = document.getElementById("api-key-input");
const saveKeyBtn = document.getElementById("save-key-btn");
const llmStatus = document.getElementById("llm-status");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarClose = document.getElementById("sidebar-close");
const sidebarOverlay = document.getElementById("sidebar-overlay");

// ═══ Markdown 渲染 ═══

if (typeof marked !== "undefined") {
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
}

function renderMarkdown(text) {
    if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
        const html = marked.parse(text);
        return DOMPurify.sanitize(html);
    }
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, "<br>");
}

// ═══ 移动端侧边栏 ═══

function openSidebar() {
    sidebar.classList.add("open");
    sidebarOverlay.classList.add("active");
}

function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("active");
}

sidebarToggle.addEventListener("click", () => {
    if (window.innerWidth <= 768) {
        openSidebar();
    } else {
        sidebar.classList.toggle("collapsed");
    }
});

sidebarClose.addEventListener("click", () => {
    if (window.innerWidth <= 768) {
        closeSidebar();
    } else {
        sidebar.classList.add("collapsed");
    }
});

sidebarOverlay.addEventListener("click", closeSidebar);

// ═══ Session 管理 ═══

let sessionId = null;

async function initSession() {
    const stored = localStorage.getItem("session_id");
    if (stored) {
        sessionId = stored;
        return;
    }
    try {
        const res = await fetch("/api/session");
        const data = await res.json();
        sessionId = data.session_id;
        localStorage.setItem("session_id", sessionId);
    } catch (err) {
        console.error("创建会话失败:", err);
    }
}

function clearSession() {
    localStorage.removeItem("session_id");
    sessionId = null;
    initSession();
}

// ═══ 工具函数 ═══

function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function getFileIcon(filename) {
    const ext = filename.split(".").pop().toLowerCase();
    const icons = { txt: "📄", md: "📝", csv: "📊", json: "📋", docx: "📘" };
    return icons[ext] || "📄";
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ═══ API Key 管理 ═══

function updateLlmStatus(enabled) {
    if (enabled) {
        llmStatus.className = "llm-status connected";
        llmStatus.querySelector(".llm-text").textContent = "已连接";
    } else {
        llmStatus.className = "llm-status disconnected";
        llmStatus.querySelector(".llm-text").textContent = "未连接";
    }
}

async function loadSettings() {
    try {
        const res = await fetch("/api/settings");
        const data = await res.json();
        updateLlmStatus(data.llm_enabled);

        const savedKey = localStorage.getItem("deepseek_api_key");
        if (savedKey && data.llm_enabled) {
            apiKeyInput.value = savedKey;
        }
    } catch (err) {
        console.error("加载设置失败:", err);
    }
}

async function saveApiKey() {
    const key = apiKeyInput.value.trim();
    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key }),
        });
        const data = await res.json();

        if (data.success) {
            updateLlmStatus(data.llm_enabled);
            if (key) {
                localStorage.setItem("deepseek_api_key", key);
            } else {
                localStorage.removeItem("deepseek_api_key");
                apiKeyInput.value = "";
            }
            addMessage("bot", data.message);
        }
    } catch (err) {
        addMessage("bot", "保存设置失败: " + err.message);
    }
}

saveKeyBtn.addEventListener("click", saveApiKey);
apiKeyInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") saveApiKey();
});

// ═══ 聊天功能 ═══

function addMessage(role, text, sources) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}-message`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "你" : "牛马";

    const content = document.createElement("div");
    content.className = "message-content";

    const textDiv = document.createElement("div");
    textDiv.className = "message-text";
    textDiv.innerHTML = renderMarkdown(text);
    content.appendChild(textDiv);

    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "message-sources";
        sourcesDiv.style.alignSelf = role === "user" ? "flex-end" : "flex-start";
        sources.forEach((s) => {
            const tag = document.createElement("span");
            tag.className = "source-tag";
            tag.innerHTML = `📎 ${escapeHtml(s.source)} <span class="source-score">${(s.score * 100).toFixed(0)}%</span>`;
            sourcesDiv.appendChild(tag);
        });
        content.appendChild(sourcesDiv);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
    return { wrapper, content, textDiv };
}

function addTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "message bot-message";
    wrapper.id = "typing-wrapper";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "牛马";

    const indicator = document.createElement("div");
    indicator.className = "typing-indicator";
    indicator.innerHTML = "<span></span><span></span><span></span>";

    wrapper.appendChild(avatar);
    wrapper.appendChild(indicator);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById("typing-wrapper");
    if (el) el.remove();
}

async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question) return;

    addMessage("user", question);
    chatInput.value = "";
    sendBtn.disabled = true;

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question, top_k: 5, session_id: sessionId || "" }),
        });

        const contentType = res.headers.get("content-type") || "";

        if (contentType.includes("text/event-stream")) {
            await handleSSEResponse(res);
        } else {
            addTypingIndicator();
            const data = await res.json();
            removeTypingIndicator();
            if (data.error) {
                addMessage("bot", "出错了：" + data.error);
            } else {
                addMessage("bot", data.answer, data.sources);
            }
        }
    } catch (err) {
        removeTypingIndicator();
        addMessage("bot", "网络错误，请检查服务器是否正在运行。");
    }

    sendBtn.disabled = false;
    chatInput.focus();
}

async function handleSSEResponse(res) {
    const { content, textDiv } = addMessage("bot", "");
    textDiv.className = "message-text stream-cursor";
    let sources = [];
    let fullText = "";

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
            const rawEvent = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);

            const lines = rawEvent.split("\n");
            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const jsonStr = line.slice(6);
                try {
                    const evt = JSON.parse(jsonStr);

                    if (evt.type === "sources") {
                        sources = evt.data || [];
                    } else if (evt.type === "delta") {
                        fullText += evt.data;
                        textDiv.innerHTML = renderMarkdown(fullText);
                        scrollToBottom();
                    } else if (evt.type === "error") {
                        textDiv.className = "message-text";
                        textDiv.innerHTML = "";
                        const errP = document.createElement("span");
                        errP.style.color = "var(--danger)";
                        errP.textContent = "错误：" + evt.data;
                        textDiv.appendChild(errP);
                        const tipP = document.createElement("div");
                        tipP.style.marginTop = "8px";
                        tipP.style.fontSize = "13px";
                        tipP.style.color = "var(--text-muted)";
                        tipP.textContent = "已切换为纯检索模式，请在侧边栏检查 API Key 设置。";
                        textDiv.appendChild(tipP);
                    } else if (evt.type === "done") {
                        textDiv.className = "message-text";
                        textDiv.innerHTML = renderMarkdown(fullText);
                        if (sources.length > 0) {
                            const sourcesDiv = document.createElement("div");
                            sourcesDiv.className = "message-sources";
                            sources.forEach((s) => {
                                const tag = document.createElement("span");
                                tag.className = "source-tag";
                                tag.innerHTML = `📎 ${escapeHtml(s.source)} <span class="source-score">${(s.score * 100).toFixed(0)}%</span>`;
                                sourcesDiv.appendChild(tag);
                            });
                            content.appendChild(sourcesDiv);
                        }
                        scrollToBottom();
                    }
                } catch (e) {
                    // 忽略解析错误
                }
            }
        }
    }

    textDiv.className = "message-text";
    if (fullText) {
        textDiv.innerHTML = renderMarkdown(fullText);
    }
}

function askExample(question) {
    chatInput.value = question;
    sendMessage();
}

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener("click", sendMessage);

// ═══ 文件上传 ═══

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", (e) => {
    handleFiles(e.target.files);
    fileInput.value = "";
});

async function handleFiles(files) {
    for (const file of files) {
        const ext = "." + file.name.split(".").pop().toLowerCase();
        const supported = [".txt", ".md", ".csv", ".json", ".docx"];
        if (!supported.includes(ext)) {
            showUploadStatus(`不支持的文件类型: ${file.name}`, "error");
            continue;
        }

        const formData = new FormData();
        formData.append("file", file);

        showUploadStatus(`正在上传 ${file.name}...`, "");

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (data.success) {
                showUploadStatus(data.message, "success");
                loadDocuments();
                loadStats();
            } else {
                showUploadStatus(data.error || "上传失败", "error");
            }
        } catch (err) {
            showUploadStatus("上传失败: " + err.message, "error");
        }
    }
}

function showUploadStatus(msg, type) {
    uploadStatus.textContent = msg;
    uploadStatus.className = "upload-status " + type;
    if (type === "success") {
        setTimeout(() => {
            uploadStatus.textContent = "";
            uploadStatus.className = "upload-status";
        }, 3000);
    }
}

// ═══ 文档列表 ═══

async function loadDocuments() {
    try {
        const res = await fetch("/api/documents");
        const data = await res.json();
        renderDocList(data.documents);
    } catch (err) {
        console.error("加载文档列表失败:", err);
    }
}

function renderDocList(docs) {
    docCount.textContent = docs.length;

    if (docs.length === 0) {
        docList.innerHTML = '<div class="doc-empty">暂无文档，请上传</div>';
        return;
    }

    docList.innerHTML = "";
    docs.forEach((doc) => {
        const item = document.createElement("div");
        item.className = "doc-item";
        item.innerHTML = `
            <span class="doc-icon">${getFileIcon(doc.name)}</span>
            <div class="doc-info">
                <div class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</div>
                <div class="doc-size">${formatSize(doc.size)}</div>
            </div>
            <button class="doc-delete" title="删除" onclick="deleteDoc('${escapeHtml(doc.name)}')">✕</button>
        `;
        docList.appendChild(item);
    });
}

async function deleteDoc(filename) {
    if (!confirm(`确定删除文件 "${filename}" 吗？`)) return;

    try {
        const res = await fetch("/api/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: filename }),
        });
        const data = await res.json();

        if (data.success) {
            showUploadStatus(data.message, "success");
            loadDocuments();
            loadStats();
        } else {
            showUploadStatus(data.error || "删除失败", "error");
        }
    } catch (err) {
        showUploadStatus("删除失败: " + err.message, "error");
    }
}

// ═══ 统计信息 ═══

async function loadStats() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();
        statDocs.textContent = data.total_documents || 0;
        statChunks.textContent = data.total_chunks || 0;
    } catch (err) {
        console.error("加载统计失败:", err);
    }
}

// ═══ 初始化 ═══

initSession();
loadSettings();
loadDocuments();
loadStats();
chatInput.focus();
