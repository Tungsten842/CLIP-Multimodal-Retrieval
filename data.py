import lancedb
import numpy as np
import torch
from torch.utils.data import Dataset


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.connect()


class CelebAPairedEmbeddings(Dataset):
    def __init__(self, db_path="dataset", split="train"):
        self.db_path = db_path
        self.split = split

        self.db = None
        self.indices_table = None
        self.images_table = None

        db_temp = lancedb.connect(db_path)
        self.length = len(db_temp.open_table(f"{split}_indices"))

        categories_table = db_temp.open_table("categories_embeddings").to_arrow()
        self.categories_embeds = torch.from_numpy(
            np.stack(categories_table["text_embeds"].to_pylist())
        ).float()

    def connect(self):
        """Method to be called by the worker_init_fn"""
        self.db = lancedb.connect(self.db_path)
        self.indices_table = self.db.open_table(f"{self.split}_indices")
        self.images_table = self.db.open_table(self.split)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pair_row = self.indices_table.to_lance().take([idx]).to_pydict()

        idx_a = pair_row["a"][0]
        idx_b = pair_row["b"][0]

        # Apply the same fix here
        img_rows = self.images_table.to_lance().take([idx_a, idx_b]).to_pydict()

        embed_a = torch.tensor(img_rows["image_embeds"][0]).float()
        embed_b = torch.tensor(img_rows["image_embeds"][1]).float()

        return (
            embed_a,
            embed_b,
            self.categories_embeds,
        )


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
