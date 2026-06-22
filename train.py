import copy
import sys
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from mlflow import MlflowClient
from mlflow.entities import Metric
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CelebAPairedEmbeddings, seed_worker
from eval import eval
from model import Model, collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"


def info_nce_loss(predicted_embeds, target_embeds, temperature):
    logits = torch.matmul(predicted_embeds, target_embeds.T) / temperature

    batch_size = predicted_embeds.shape[0]
    labels = torch.arange(batch_size, device=predicted_embeds.device)

    loss = F.cross_entropy(logits, labels)

    return loss


def run_step(model, batch):
    imga, imgb, pos_txt, pos_msk, neg_txt, neg_msk = [
        b.to(device, non_blocking=True) for b in batch
    ]
    predicted_imgb = model(imga, pos_txt, pos_msk, neg_txt, neg_msk)
    nce_loss = info_nce_loss(predicted_imgb, imgb, 0.07)
    cos_dist = 1.0 - (predicted_imgb * imgb).sum(dim=-1).mean()
    return nce_loss, cos_dist


def validate(model, val_loader):
    model.eval()
    val_losses, val_cos = [], []
    with torch.no_grad():
        for batch in val_loader:
            nce, cos = run_step(model, batch)
            val_losses.append(nce.item())
            val_cos.append(cos.item())
    model.train()
    return sum(val_losses) / len(val_losses), sum(val_cos) / len(val_cos)


def main():
    config = {
        "learning_rate": 1e-4,
        "max_lr": 1e-3,
        "batch_size": 1024,
        "embed_dim": 512,
        "nhead": 8,
        "dataset_percentage": 0.5,
        "seed": 42,
    }
    if len(sys.argv) == 2:
        config["dataset_percentage"] = float(sys.argv[1])

    g_train = torch.Generator()
    g_val = torch.Generator()
    g_train.manual_seed(config["seed"])
    g_val.manual_seed(config["seed"])

    train_dataset = CelebAPairedEmbeddings(
        "train", config["seed"], config["dataset_percentage"]
    )
    val_dataset = CelebAPairedEmbeddings(
        "valid", config["seed"], config["dataset_percentage"]
    )
    model = Model(embed_dim=config["embed_dim"], nhead=config["nhead"]).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g_train,
        multiprocessing_context="spawn",
    )
    config["val_interval"] = max(1, len(train_loader) // 16) * config["batch_size"]
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g_val,
        multiprocessing_context="spawn",
        prefetch_factor=32,
    )

    optimizer = Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=1,
    )

    model.train()
    step = 0

    mlflow.config.enable_async_logging()
    client = MlflowClient()
    mlflow.set_experiment("compositional-transformer")

    last_step = (len(train_loader) - 1) * config["batch_size"]
    metric_buffer = []
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(config)

        best_loss = float("inf")
        val_loss = 0
        pbar = tqdm(train_loader, unit_scale=config["batch_size"])
        for batch in pbar:
            optimizer.zero_grad()
            nce_loss, cos_distance = run_step(model, batch)
            loss = nce_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            timestep = int(time.time() * 1000)
            metric_buffer.extend(
                [
                    Metric("nce_loss", nce_loss, timestep, step),
                    Metric("cos_distance", cos_distance, timestep, step),
                ]
            )

            is_log_interval = step % (32 * config["batch_size"]) == 0 and step > 0
            is_val_interval = step % config["val_interval"] == 0 and step > 0

            if is_val_interval or (step >= last_step):
                val_loss, val_cos = validate(model, val_loader)
                mlflow.log_metrics(
                    {"val_nce_loss": val_loss, "val_cos_distance": val_cos}, step=step
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(model.state_dict())

            if is_log_interval or (step >= last_step):
                client.log_batch(run_id=run_id, metrics=metric_buffer)
                metric_buffer.clear()
                pbar.set_postfix(
                    {"train_loss": f"{loss:.6f}", "val_loss": f"{val_loss:.6f}"}
                )

            step += config["batch_size"]

        model.load_state_dict(best_model)
        precision, recall = eval("model", model)

        mlflow.log_metrics({"precision": precision, "recall": recall})
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
            code_paths=list(Path(".").glob("*.py")),
        )


if __name__ == "__main__":
    main()
