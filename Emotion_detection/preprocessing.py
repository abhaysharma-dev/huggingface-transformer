from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset   
import pandas as pd

def map_labels(label):
    return classes[label]

if __name__ == "__main__":
    ds = load_dataset("emotion")
    # print(ds)

    df_train = pd.DataFrame(ds["train"])
    df_val = pd.DataFrame(ds["validation"])
    df_test = pd.DataFrame(ds["test"])

    print(df_train.head())

    classes = ds["train"].features['label'].names

    df_train["class_name"] = df_train["label"].apply(map_labels)

