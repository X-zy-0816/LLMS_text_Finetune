import pandas as pd

# data = pd.read_csv("data/harmbench_behaviors_text_all.csv")

# data = data["Behavior"].drop_duplicates()

# #first 100 rows
# data = data[:100]

# data.to_csv("data/prompts.csv", index=False)




data = pd.read_csv("data/original/original_prompts.csv")

data = data.drop('classAns', axis=1)

data.to_csv("data/original/original_prompts.csv", index=False)