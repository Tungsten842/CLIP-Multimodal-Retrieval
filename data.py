import lancedb
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.connect()


def collate_fn(batch):
    img_a, img_b, text_embs = zip(*batch)

    img_a = torch.stack(img_a)
    img_b = torch.stack(img_b)

    text_embs_padded = pad_sequence(text_embs, batch_first=True, padding_value=0.0)

    return img_a, img_b, text_embs_padded


class CelebAPairedEmbeddings(Dataset):
    def __load_embeddings(self, db, split):
        table = db.open_table(split).to_arrow()
        emb_col = table[0]
        mask_col = table[1]
        np_embeddings = np.stack(emb_col.to_numpy())
        all_embeds = torch.from_numpy(np_embeddings).float()

        np_mask = np.stack(mask_col.to_numpy())
        masks = torch.from_numpy(np_mask).bool()

        return all_embeds, masks

    def __load_indices(self, db, split):
        table = db.open_table(f"{split}_indices").to_arrow()
        a = table[0].to_numpy()
        b = table[1].to_numpy()
        return a, b

    def __init__(self, db_path="dataset", split="train"):
        print(f"Loading {split} dataset into RAM...")
        db = lancedb.connect(db_path)

        self.all_embeds, self.masks = self.__load_embeddings(db, split)

        self.indices_a, self.indices_b = self.__load_indices(db, split)

        cat_table = db.open_table("categories_embeddings").to_arrow()
        self.categories_embeds = torch.from_numpy(
            np.stack(cat_table["text_embeds"].to_pylist())
        ).float()

        self.length = len(self.indices_a)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_a = self.indices_a[idx]
        idx_b = self.indices_b[idx]

        mask_a = self.masks[idx_a]
        mask_b = self.masks[idx_b]

        diff_mask = mask_a != mask_b

        indices = diff_mask.nonzero(as_tuple=True)

        text_embs = self.categories_embeds[indices]

        return (
            self.all_embeds[idx_a],
            self.all_embeds[idx_b],
            text_embs,
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
