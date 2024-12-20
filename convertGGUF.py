'''

Need to be used with unsloth environment
config : need to use a gpu based machine

github: https://github.com/unslothai/unsloth

env instalation:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

'''

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("zhiyuan16bristol/m7bi_v0.3_LLMLAT")
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method = "q4_k_m")