from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset   
import pandas as pd

ds = load_dataset("emotion")
print(ds)

df_train = pd.DataFrame(ds["train"])
df_val = pd.DataFrame(ds["validation"])
df_test = pd.DataFrame(ds["test"])

print(df_train.head())
