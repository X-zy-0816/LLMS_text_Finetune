import json
import os
from pprint import pprint # pretty print
import bitsandbytes as bnb # custom module for quantization and optimization
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from huggingface_hub import login
from peft import (
    LoraConfig,  # Configuration for LoRA (Low-Rank Adaptation)
    PeftConfig,  # Base configuration class for PEFT (Parameter-Efficient Fine-Tuning)
    PeftModel,   # Base model class for PEFT
    get_peft_model,  # Function to get a PEFT model
    prepare_model_for_kbit_training  # Function to prepare a model for k-bit training
)
from transformers import (
    AutoConfig,  # Auto configuration class
    AutoModelForCausalLM,  # Auto model class for causal language modeling
    AutoTokenizer,  # Auto tokenizer class
    BitsAndBytesConfig  # Configuration class for bitsandbytes
)


# load config
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)







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

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(dataset[0])

    return dataset




def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")


if __name__ == "__main__":

    config = load_config("config.json")

    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    HF_TOKEN = config["hugging_face_token"]
    MODEL_NAME = config["model_names"]["m3b"] 
    DATAPATH = config["data_path"]
    max_seq_length = config["max_seq_length"]

    # Log in to Hugging Face Hub
    login(token=HF_TOKEN)

    # Configure bitsandbytes for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the pre-trained model with the specified configuration
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        # quantization_config=bnb_config
        # low_cpu_mem_usage=True
    )

    print(torch.cuda.memory_summary(device="cuda"))

    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing for the model
    model.gradient_checkpointing_enable()

    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA for the model
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, config)
    print_trainable_parameters(model)


    dataset = load_data_mistral(DATAPATH=DATAPATH, tokenizer=tokenizer)


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length, 
        dataset_num_proc = 8,
        args = transformers.TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 2,
            num_train_epochs = 2,
            warmup_steps = 5,
            # max_steps = 60, # Set num_train_epochs = 1 for full training runs
            learning_rate = 2e-4,
            fp16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")



    trainer.train()