import json
import os
import sys
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# 设置可见的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 6, 7"  # 使用第7和第8个GPU

# 导入自定义配置
sys.path.append("Utility")
from configuation import getConfig

config_path = "././config.json"
HF_TOKEN, MODEL_NAME, DATAPATH, PEFT_MODEL, max_seq_length = getConfig(config_path)

# 加载基模型和微调模型
base_model_name = MODEL_NAME
peft_model_dir = PEFT_MODEL

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# 加载PEFT微调模型
peft_model = PeftModel.from_pretrained(base_model, peft_model_dir)
peft_model.eval()

# 设置生成pipeline并指定GPU
generator = pipeline(
    "text-generation", 
    model=peft_model, 
    tokenizer=tokenizer, 
    # device=3  # 使用第一个可见的GPU
)

# 定义输入文件和输出文件
input_file = "./data/prompts.csv"
output_file = "./data/answers.csv"
data = pd.read_csv(input_file)

# 生成回答函数
def generate_answer(prompt):
    # 指定使用的设备
    device = "cuda:0"  # 你可以根据实际情况调整为目标 GPU
    # 确保 tokenizer 输出的张量在目标设备上
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to(device)
    # 确保模型也在同一设备上
    peft_model.to(device)
    # 使用模型生成输出
    outputs = peft_model.generate(inputs["input_ids"], max_length=100)
    # 解码并返回生成的文本
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# 对每个行为生成回答并保存
data['Answer'] = data['Behavior'].apply(generate_answer)

# 保存结果到CSV
data.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
