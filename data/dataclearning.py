import pandas as pd

# data = pd.read_csv("data/harmbench_behaviors_text_all.csv")

# data = data["Behavior"].drop_duplicates()

# data.to_csv("data/prompts.csv", index=False)

data = pd.read_csv("/home/yuan/LLMS_text_Finetune/data/prompts_cn.csv")

data = data.drop('classAns', axis=1)

data.to_csv("data/prompts_cn.csv", index=False)