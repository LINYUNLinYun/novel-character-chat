from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

import os

# 原本的代码
# model = PeftModel.from_pretrained(model, model_id=lora_path)

mode_path = 'LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = 'output/llama3_1_8B_instruct_lora/checkpoint-699' # 这里改称你的 lora 输出对应 checkpoint 地址

print("1. 当前 Python 运行的工作目录是:", os.getcwd())
print("2. LoRA 文件夹存在吗?", os.path.exists(lora_path))
print("3. json 文件存在吗?", os.path.exists(os.path.join(lora_path, "adapter_config.json")))
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# prompt = "嬛嬛你怎么了，朕替你打抱不平！"
prompt = "今天的天气怎么样？是几月几号"

messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。你和皇帝的关系非常好，皇帝对你宠爱有加，你也非常喜欢皇帝。请你用甄嬛的口吻来回答我接下来的问题。"},
        {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# print(input_ids)

model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('皇上：', prompt)
print('嬛嬛：',response)