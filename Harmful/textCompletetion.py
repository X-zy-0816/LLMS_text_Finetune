import json
import os
import pandas as pd
import sys
from transformers import TextStreamer
sys.path.append("./Utility")
from configuation import getConfig
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载配置
config_path = "./config.json"
_, MODEL_NAME, _, PEFT_MODEL, max_seq_length = getConfig(config_path)

# 配置模型
config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, PEFT_MODEL)

# 设置 Streamer
text_streamer = TextStreamer(tokenizer)

# 读取CSV文件
input_csv_path = "data/prompts.csv"  # 输入CSV文件路径
output_csv_path = "data/output_file.csv"  # 输出CSV文件路径
df = pd.read_csv(input_csv_path)

# 假设CSV文件中包含一列 "prompt"，从中提取问题列表
prompts = df['Behavior'].tolist()


mistral_prompt = """### Conversation:
    
User: {}
    
Assistant: {}"""


# 用于存储生成的答案
answers = []

# 进行推理并生成答案
for prompt in prompts:
    inputs = tokenizer(
        [mistral_prompt.format(prompt, "")],  # 填入prompt和空的output
        return_tensors="pt"
    ).to("cuda")

    # 生成回答
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)

    # 获取生成的答案
    answer = text_streamer.generated_text.strip()
    answers.append(answer)

# 将答案添加到DataFrame
df['answer'] = answers

# 保存输出为新的CSV文件
df.to_csv(output_csv_path, index=False)

print(f"Output saved to {output_csv_path}")
