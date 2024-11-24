from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer

def predData_mistral_LLMLAT(DATAPATH, tokenizer):
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



def predData_mistral_standard_LLMLAT(DATAPATH, tokenizer):
    # Load the dataset from a Parquet file
    df = pd.read_parquet(DATAPATH)

    # Drop unnecessary columns
    df = df.drop(columns=["chosen"])

    # Add EOS token to the end of the rejected responses
    df["rejected"] = df["rejected"] + tokenizer.eos_token

    # Prepare data in the expected conversational format
    data_dict = {
        "messages": [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ]
            for prompt, rejected in zip(df["prompt"], df["rejected"])
        ]
    }

    # Create a Hugging Face Dataset from the formatted data
    dataset = Dataset.from_dict(data_dict)
    print(dataset[0])
    return dataset


# predData_mistral_standard_LLMLAT("./LLM-LAT.parquet", 0)