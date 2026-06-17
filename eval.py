import lancedb
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPTokenizer

device = "cuda"


def evaluate_retrieval(
    retrieved_indices: list[int], ground_truth_indices: list[int], k: int
):
    """
    Evaluate the retrieval performance for a single source image.

    Args:
    ----
        retrieved_indices: list of image IDs predicted by the model,
            ordered by similarity (descending).
        ground_truth_indices: list of valid target IDs from the benchmark JSON.
        k: the cutoff for top-K evaluation (e.g., 1, 5, 10).

    Return:
    ------
        A dictionary containing Recall@K and Precision@K.

    """
    # Isolate the top K predictions
    top_k_retrieved = retrieved_indices[:k]

    # Calculate the intersection between predictions and ground truth
    hits = set(top_k_retrieved).intersection(set(ground_truth_indices))
    num_hits = len(hits)

    # Metrics calculations
    # Recall@K (Hit Rate): 1 if at least one match is found, 0 otherwise
    recall_at_k = 1 if num_hits > 0 else 0

    # Precision@K: Fraction of top K predictions that are correct
    precision_at_k = num_hits / k

    return precision_at_k, recall_at_k


def construct_query(query):
    tmp = query.split(", ")
    txt = ""
    for i, q in enumerate(tmp):
        category = q[1:].replace("_", " ")
        sign = "" if q[0] == "+" else "not "
        if i > 0:
            txt += "and "
        txt += sign + category + " "
    return txt


def evaluate_dataset(tokenizer, model, image_embs, eval_records, k=10):
    image_embs = normalize(image_embs)

    mean_precision = 0.0
    mean_recall = 0.0
    total_samples = 0
    for entry in eval_records:
        print("Query: " + entry["query"])
        query_text = construct_query(entry["query"])
        print("Text: " + query_text)

        tokens = tokenizer(query_text, padding=True, return_tensors="pt").to(device)
        text_embs = model.get_text_features(**tokens).pooler_output
        text_embs = normalize(text_embs)

        ground_truth_dict = entry["ground_truth"]

        values = list(ground_truth_dict.values())

        source_img_idx = list(map(int, ground_truth_dict.keys()))
        source_idx = torch.tensor(source_img_idx, device=device)

        source_img_embs = image_embs[source_img_idx]

        query_embs = source_img_embs + text_embs * 2.5
        query_embs = normalize(query_embs)

        sims = query_embs @ image_embs.T

        # Remove source images
        row_indices = torch.arange(len(source_idx), device=device)
        sims[row_indices, source_idx] = -float("inf")

        results = torch.topk(sims, k, largest=True)[1]
        results = results.tolist()

        query_precision = 0.0
        query_recall = 0.0
        for retrieved, val_list in zip(results, values):
            precision, recall = evaluate_retrieval(retrieved, val_list, k)
            query_precision += precision
            query_recall += recall

        query_len = len(ground_truth_dict.items())
        print(f"Query precision: {query_precision * 100 / query_len:.2f}%")
        print(f"Query recall: {query_recall * 100 / query_len:.2f}%")
        mean_precision += query_precision
        mean_recall += query_recall
        total_samples += query_len
    mean_precision = mean_precision / total_samples
    mean_recall = mean_recall / total_samples
    return mean_precision, mean_recall


def main():
    model_id = "openai/clip-vit-base-patch32"
    with torch.inference_mode():
        eval_records = pd.read_json("celeba_evaluation.json").to_dict("records")

        db = lancedb.connect("dataset")
        table = db.open_table("test")
        col = table.to_lance().to_table(columns=["image_embeds"])["image_embeds"]
        image_embs = torch.from_numpy(np.stack(col.to_numpy())).to(device)

        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)

        precision, recall = evaluate_dataset(tokenizer, model, image_embs, eval_records)

        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")


if __name__ == "__main__":
    main()
