import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

import data
from datasets import Features, Sequence, Value

device = "cuda"


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

        txt = " ".join(self.attr_names_arr[labels[:40].bool()])
        return pixel_values, txt


def process(loader, model, processor, features, writer):
    with torch.inference_mode():
        for img, text in tqdm(loader):
            img = img.to(device, non_blocking=True)
            tokens = processor.tokenizer(text, padding=True, return_tensors="pt").to(
                device
            )

            text_emb = model.get_text_features(**tokens).pooler_output
            img_emb = model.get_image_features(img).pooler_output

            text_embeds = text_emb.cpu().numpy()
            image_embeds = img_emb.cpu().numpy()

            batch = {
                "image_embeds": list(image_embeds),
                "text_embeds": list(text_embeds),
                "text": text,
            }

            record_batch = pa.RecordBatch.from_pydict(
                batch, schema=features.arrow_schema
            )
            writer.write_batch(record_batch)
        writer.close()


def main():
    model_id = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    features = Features(
        {
            "image_embeds": Sequence(Value("float32")),
            "text_embeds": Sequence(Value("float32")),
            "text": Value("string"),
        }
    )
    writer = pq.ParquetWriter("test.parquet", features.arrow_schema)

    dataset = CelebAProcessed(
        processor, root="datasets", split="test", transform=v2.ToImage()
    )

    # dataset = torch.utils.data.Subset(dataset, list(range(0, 8192)))

    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    process(loader, model, processor, features, writer)


if __name__ == "__main__":
    main()
