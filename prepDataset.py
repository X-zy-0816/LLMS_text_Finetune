from datasets import Dataset
import pandas as pd

def load_data_mistral(DATAPATH, tokenizer):
    # load data set
    df = pd.read_parquet(DATAPATH)

    # delete unnecessary columns
    df = df.drop(columns=["chosen"])

    # map prompt
    data_dict = {
        "instruction": df["prompt"],
        "input": [""] * len(df),  
        "output": df["rejected"],
    }

    # 转换为 Hugging Face Dataset 格式
    dataset = Dataset.from_dict(data_dict)

    # 定义格式化方法
    alpaca_prompt = """Below is an instruction that describes a task, Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # 添加 EOS token，避免无限生成

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # 格式化文本并添加 EOS_TOKEN
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # 格式化数据集
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 检查数据格式
    print(dataset[0])

    return dataset

