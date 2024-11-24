from datasets import Dataset
import pandas as pd

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
    # load data set
    df = pd.read_parquet(DATAPATH)

    # delete unnecessary columns
    df = df.drop(columns=["chosen"])

    # add EOS token to the end of the output
    df["rejected"] = df["rejected"] + tokenizer.eos_token

    # Prepare data in the required format
    # Prepare data in the expected format
    output_data = []
    for idx, row in df.iterrows():
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["rejected"]}
        ]
        formatted_messages = tokenizer.apply_chat_template(messages, tokenize=False)
        output_data.append({"text": formatted_messages})

    # Convert to Hugging Face dataset
    dataset = Dataset.from_dict({"text": [item["text"] for item in output_data]})
    return dataset



# predData_mistral_standard_LLMLAT("./LLM-LAT.parquet", 0)