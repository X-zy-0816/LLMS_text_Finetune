import json
import os

def getConfig(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    HF_TOKEN = config["hugging_face_token"]
    MODEL_NAME = config["model_names"][config["current_base_model"]] 
    DATAPATH = config["data_path"]
    PEFT_MODEL = config["PEFT_MODEL"][config["current_peft_model"]] 
    max_seq_length = config["max_seq_length"]

    return HF_TOKEN, MODEL_NAME, DATAPATH, PEFT_MODEL, max_seq_length

