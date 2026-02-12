from datasets import load_dataset
import pandas as pd

dataset = load_dataset("deepset/prompt-injections")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

print(train_df.head())
print(f"\nLabel distribution:\n{train_df['label'].value_counts()}")

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

