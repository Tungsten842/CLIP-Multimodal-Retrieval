import numpy as np
import pandas
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Features, Sequence, Value
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


class CelebAProcessed(CelebA):
    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.attr_names_arr = np.array(self.attr_names)[:40]

    def __getitem__(self, index):
        img, labels = super().__getitem__(index)
        pixel_values = self.processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        categories_mask = labels[:40].bool()
        return pixel_values, categories_mask


def proces_categories_embeddings(dataset, processor, model):
    attr_names = [attr.replace("_", " ") for attr in dataset.attr_names[:40]]

    tokens = processor.tokenizer(
        attr_names, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        text_outputs = model.get_text_features(**tokens).pooler_output
        text_embeds = text_outputs.cpu().numpy()

    features = Features({"text_embeds": Sequence(Value("float32"))})

    table = pa.Table.from_pydict(
        {"text_embeds": text_embeds.tolist()}, schema=features.arrow_schema
    )
    pq.write_table(table, "categories_embeddings.parquet")


def process(loader, model, processor, split):
    features = Features(
        {
            "image_embeds": Sequence(Value("float32")),
            "category_masks": Sequence(Value("bool")),
        }
    )

    writer = pq.ParquetWriter("dataset/" + split + ".parquet", features.arrow_schema)

    with torch.inference_mode():
        for img, masks in tqdm(loader):
            img = img.to(device, non_blocking=True)

            img_emb = model.get_image_features(img).pooler_output
            image_embeds = img_emb.cpu().numpy()

            category_masks = masks.numpy()

            batch = {
                "image_embeds": list(image_embeds),
                "category_masks": list(category_masks),
            }

            record_batch = pa.RecordBatch.from_pydict(
                batch, schema=features.arrow_schema
            )
            writer.write_batch(record_batch)
        writer.close()


def generate_dataset(split):
    df = pandas.read_parquet(
        "dataset/" + split + ".parquet", columns=["category_masks"]
    )
    np_mask = np.stack(df["category_masks"].values)
    masks = torch.from_numpy(np_mask).to(device).half()

    num_masks = masks.size(0)
    mask_sums = masks.sum(dim=1, keepdim=True).half()

    schema = pa.schema([("a", pa.uint32()), ("b", pa.uint32())])
    writer = pq.ParquetWriter("dataset/" + split + "_indices.parquet", schema)

    batch_size = 2048
    for i in tqdm(range(0, num_masks, batch_size)):
        end = min(i + batch_size, num_masks)
        batch = masks[i:end]

        # Hamming distance
        dot = torch.matmul(batch, masks.T)
        dists = mask_sums[i:end] + mask_sums.T - 2 * dot

        # Pick indices with hamming distance < 4
        matches = (dists < 4).nonzero()
        # Correct index
        matches[:, 0] += i
        # Remove self matches
        matches = matches[matches[:, 0] != matches[:, 1]]

        matches = matches.to(torch.uint32).cpu().numpy()
        a, b = matches[:, 0], matches[:, 1]
        table = pa.Table.from_arrays([a, b], names=["a", "b"])
        writer.write_table(table)
    writer.close()


def main():
    model_id = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    for split in ["train", "valid", "test"]:
        dataset = CelebAProcessed(processor, root="dataset", split=split)
        loader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )

        process(loader, model, processor, split)

        generate_dataset(split)

        if split == "train":
            proces_categories_embeddings(dataset, processor, model)


if __name__ == "__main__":
    main()
