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
    # Load dataset from the Parquet file
    df = pd.read_parquet(DATAPATH)

    # Drop unnecessary columns
    df = df.drop(columns=["chosen"])

    # Add EOS token to the rejected responses
    EOS_TOKEN = tokenizer.eos_token
    df["rejected"] = df["rejected"] + EOS_TOKEN

    # Create a dictionary to match the chat format
    data_dict = {
        "prompt": df["prompt"],
        "response": df["rejected"]
    }

    dataset = Dataset.from_dict(data_dict)

    # Define the chat template for formatting
    mistral_prompt = """### Conversation:
    
    User: {}
    
    Assistant: {}"""

    def formatting_chat_func(examples):
        prompts = examples["prompt"]
        responses = examples["response"]
        texts = []
        for prompt, response in zip(prompts, responses):
            # Format the conversation with the provided chat template
            text = mistral_prompt.format(prompt, response)
            texts.append(text)
        return {"text": texts}

    # Apply the formatting function to the dataset
    dataset = dataset.map(formatting_chat_func, batched=True)

    # Print a sample to verify the format
    print(dataset['text'][1])

    return dataset


# predData_mistral_standard_LLMLAT("./LLM-LAT.parquet", 0)