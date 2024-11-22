import json
import os


config_path = "./config.json"
with open(config_path, "r") as f:
    config_file = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
PEFT_MODEL = config_file["PEFT_MODEL"]


from peft import (
    PeftConfig,  # Base configuration class for PEFT (Parameter-Efficient Fine-Tuning)
    PeftModel,   # Base model class for PEFT
    get_peft_model,  # Function to get a PEFT model
    prepare_model_for_kbit_training  # Function to prepare a model for k-bit training
)

from transformers import (
    AutoModelForCausalLM,  # Auto model class for causal language modeling
    AutoTokenizer,  # Auto tokenizer class
)

config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    #quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)