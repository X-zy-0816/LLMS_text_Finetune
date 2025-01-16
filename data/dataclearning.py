import pandas as pd

data = pd.read_csv("data/harmbench_behaviors_text_all.csv")

data = data["Behavior"].drop_duplicates()

data.to_csv("data/prompts.csv", index=False)