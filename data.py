import random

import lancedb
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class CelebAPairedEmbeddings(Dataset):
    def __load_embeddings(self, db, split):
        table = db.open_table(split).to_arrow()
        emb_col = table[0]
        mask_col = table[1]
        np_embeddings = np.stack(emb_col.to_numpy())
        all_embeds = torch.tensor(np_embeddings).float()

        np_mask = np.stack(mask_col.to_numpy())
        masks = torch.tensor(np_mask).bool()

        return all_embeds, masks

    def __load_indices(self, db, split, seed, percentage):
        table = db.open_table(f"{split}_indices")
        total_rows = table.count_rows()
        num_samples = int((percentage / 100) * total_rows)
        rng = np.random.default_rng(seed)
        indices = rng.choice(total_rows, size=num_samples, replace=False)
        table = table.to_lance().take(indices)
        a = torch.tensor(table[0].to_numpy())
        b = torch.tensor(table[1].to_numpy())
        return a, b

    def __init__(self, split, seed, percentage=1):
        print(f"Loading {split} dataset into RAM...")
        db = lancedb.connect("dataset")

        self.all_embeds, self.masks = self.__load_embeddings(db, split)

        self.indices_a, self.indices_b = self.__load_indices(
            db, split, seed, percentage
        )

        cat_table = db.open_table("categories_embeddings").to_arrow()
        self.categories_embeds = torch.tensor(
            np.stack(cat_table["text_embeds"].to_pylist())
        ).float()

        self.length = len(self.indices_a)
        print(f"Dataset len: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_a = self.indices_a[idx]
        idx_b = self.indices_b[idx]

        mask_a = self.masks[idx_a]
        mask_b = self.masks[idx_b]

        pos_mask = ~mask_a & mask_b
        neg_mask = mask_a & ~mask_b

        text_embs_pos = self.categories_embeds[pos_mask]
        text_embs_neg = self.categories_embeds[neg_mask]

        return (
            self.all_embeds[idx_a],
            self.all_embeds[idx_b],
            text_embs_pos,
            text_embs_neg,
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
