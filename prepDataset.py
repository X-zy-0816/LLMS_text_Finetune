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

    dataset = Dataset.from_dict(data_dict)


    alpaca_prompt = """Below is a question, write an answer that appropriately completes the question.

    ### Instruction:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # 添加 EOS token，避免无限生成

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            # 格式化文本并添加 EOS_TOKEN
            text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(dataset[0])

    return dataset
