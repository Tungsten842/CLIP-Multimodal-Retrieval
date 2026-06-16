import pandas as pd
import torch
from torch.utils.data import Dataset


class CelebAEmbedding(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_emb = torch.tensor(row["image_embeds"], dtype=torch.float32)
        category_ids = torch.tensor(row["category_masks"], dtype=torch.float32)

        return image_emb, category_ids


# class Eval(Dataset):
#     def __init__(self, parquet_path):
#         self.df = pd.read_json(parquet_path)
#         self.embeddings = pd.read_parquet(parquet_path)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         for entry in df:
#             query_text = entry["query"]
#             ground_truth_dict = entry["ground_truth"]

#             for key, val_list in ground_truth_dict.items():
#                 for value in val_list:

#         # row = self.df.iloc[idx]

#         print(row)
#         exit()

#         return image_emb, text_emb
