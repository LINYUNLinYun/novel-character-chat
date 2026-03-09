import os

import torch

try:
    import gradio as gr
except ModuleNotFoundError:
    raise SystemExit("缺少依赖: gradio。请先执行: python3 -m pip install -r requirement.txt")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
except ModuleNotFoundError:
    raise SystemExit("缺少依赖: transformers。请先执行: python3 -m pip install -r requirement.txt")

try:
    from peft import PeftModel
except ModuleNotFoundError:
    raise SystemExit("缺少依赖: peft。请先执行: python3 -m pip install -r requirement.txt")
from threading import Thread

# ================= 1. 定义绝对路径 =================
# 强烈建议使用绝对路径！
mode_path = 'LLM-Research/Meta-Llama-3___1-8B-Instruct' 
lora_path = 'output/llama3_1_8B_instruct_lora/checkpoint-699' 

if not os.path.exists(mode_path):
    raise SystemExit(f"基础模型路径不存在: {mode_path}")

if not os.path.exists(lora_path):
    raise SystemExit(f"LoRA 路径不存在: {lora_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ================= 2. 加载模型与分词器 =================
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

print("正在加载基础模型(这可能需要一点时间)...")
model = AutoModelForCausalLM.from_pretrained(
    mode_path, 
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=dtype,
    trust_remote_code=True
).eval()

print("正在挂载甄嬛(LoRA)灵魂...")
model = PeftModel.from_pretrained(model, model_id=lora_path)
print("加载完成！准备启动 WebUI...")

# ================= 3. 核心：对话与推理函数 =================
def chat_generator(message, history):
    """
    message: 用户当前输入的新消息 (字符串)
    history: Gradio 自动维护的历史记录，格式为 [[用户话1, 甄嬛话1], [用户话2, 甄嬛话2]]
    """
    
    # 3.1 构建包含历史记录的 messages 列表
    # 首先放入系统设定 (System Prompt)
    messages =[
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。请用甄嬛的语气、口吻和自称（如臣妾）来回答皇上的问题。"}
    ]
    
    # 接着，把之前的历史记录一段段塞进去
    # 兼容 Gradio 新旧两种 history 结构：
    # 1) 旧版: [[user, assistant], ...]
    # 2) 新版: [{"role": "user"/"assistant", "content": ...}, ...]
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content is not None:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg is not None:
                messages.append({"role": "user", "content": str(user_msg)})
            if bot_msg is not None:
                messages.append({"role": "assistant", "content": str(bot_msg)})
        
    # 最后，放入用户当前刚发的最新消息
    messages.append({"role": "user", "content": message})

    # 3.2 使用 Llama-3 模板将消息转为 Token IDs
    model_inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, # 这里直接转为 tensor
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(model_inputs)

    # 3.3 设置流式输出器 (打字机效果的核心)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 3.4 配置生成参数
    generation_kwargs = dict(
        inputs=model_inputs,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7, # 稍微带点随机性，回答更生动
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 3.5 把生成任务放到后台线程运行，以免阻塞主线程的 UI 刷新
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 3.6 从流式输出器中逐字读取并返回给前端
    response = ""
    for new_text in streamer:
        response += new_text
        yield response # yield 是 Python 生成器用法，Gradio 靠它实现打字机效果

# ================= 4. 构建并启动 Gradio 界面 =================
# Gradio 提供的极简聊天界面封装
demo = gr.ChatInterface(
    fn=chat_generator,
    title="🌸 甄嬛陪聊专属大模型 🌸",
    description="我是你的嬛嬛，皇上今天想聊点什么？",
    chatbot=gr.Chatbot(height=500),
)

if __name__ == "__main__":
    # 启动 Web 服务
    # server_name="0.0.0.0" 允许局域网/公网通过 IP 访问
    # 默认关闭 share，避免 frpc 下载失败导致的报错；需要公网链接时设置 GRADIO_SHARE=1
    share_enabled = os.getenv("GRADIO_SHARE", "0") == "1"
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=share_enabled)