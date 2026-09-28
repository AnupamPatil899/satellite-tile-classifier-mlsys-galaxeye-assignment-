import os
import time
import pandas as pd
from PIL import Image
from app.classifier import classifier
from app.config import settings

def run_evaluation():
    dataset_dir = os.path.join(
        settings.BASE_DIR,
        "Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles",
        "be-mlsys-assignment-dataset"
    )
    eval_set_dir = os.path.join(dataset_dir, "eval_set")
    labels_csv_path = os.path.join(dataset_dir, "eval_labels.csv")

    if not os.path.exists(labels_csv_path) or not os.path.exists(eval_set_dir):
        print(f"Error: Eval dataset or labels not found at {dataset_dir}")
        return

    df_ground_truth = pd.read_csv(labels_csv_path)
    print(f"Loaded {len(df_ground_truth)} evaluation ground truth labels.")
    print("Classes in evaluation set:", sorted(df_ground_truth["true_label"].unique()))
    print(f"Active Inference Engine: {classifier.runtime_engine}")

    results = []
    latencies = []

    print("\nRunning offline inference across evaluation set...")
    start_all = time.time()

    for idx, row in df_ground_truth.iterrows():
        fname = row["filename"]
        true_label = row["true_label"]
        img_path = os.path.join(eval_set_dir, fname)

        if not os.path.exists(img_path):
            print(f"Warning: Tile {img_path} not found. Skipping.")
            continue

        img = Image.open(img_path)
        pred = classifier.predict(img)

        latencies.append(pred["inference_latency_ms"])
        results.append({
            "filename": fname,
            "true_label": true_label,
            "pred_label": pred["predicted_class"],
            "confidence": pred["confidence"],
            "margin": pred["margin"],
            "entropy": pred["entropy"],
            "status": pred["status"],
            "latency_ms": pred["inference_latency_ms"],
            "is_correct": (true_label == pred["predicted_class"])
        })

    total_time = time.time() - start_all
    eval_df = pd.DataFrame(results)

    total = len(eval_df)
    correct = eval_df["is_correct"].sum()
    accuracy = correct / total if total > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    uncertain_count = (eval_df["status"] == "UNCERTAIN").sum()

    print("\n" + "=" * 65)
    print("        GALAXEYE OFFLINE CLASSIFIER EVALUATION REPORT")
    print("=" * 65)
    print(f"Active Inference Engine:      {classifier.runtime_engine}")
    print(f"Total Evaluation Tiles:       {total}")
    print(f"Correct Predictions:          {correct}/{total}")
    print(f"Overall Accuracy:             {accuracy * 100:.2f}%")
    print(f"Error Rate:                   {(1 - accuracy) * 100:.2f}%")
    print(f"Mean CPU Inference Latency:   {avg_latency:.2f} ms/tile")
    print(f"Total Evaluation Elapsed:     {total_time:.2f} s")
    print(f"Mean Shannon Entropy:         {eval_df['entropy'].mean():.3f} [0, 1]")
    print(f"Flagged as UNCERTAIN:         {uncertain_count} tiles ({uncertain_count/total*100:.1f}%)")
    print("=" * 65)

    print("\nPer-Class Performance Metrics:")
    classes = sorted(df_ground_truth["true_label"].unique())
    class_stats = []
    for c in classes:
        sub = eval_df[eval_df["true_label"] == c]
        pred_sub = eval_df[eval_df["pred_label"] == c]
        true_pos = eval_df[(eval_df["true_label"] == c) & (eval_df["pred_label"] == c)]
        
        recall = len(true_pos) / len(sub) if len(sub) > 0 else 0.0
        precision = len(true_pos) / len(pred_sub) if len(pred_sub) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        class_stats.append({
            "Class": c,
            "Support": len(sub),
            "Precision": f"{precision:.2f}",
            "Recall": f"{recall:.2f}",
            "F1-Score": f"{f1:.2f}"
        })

    report_df = pd.DataFrame(class_stats)
    print(report_df.to_string(index=False))

    print("\nConfusion Matrix (Rows = Ground Truth, Columns = Predicted):")
    conf_matrix = pd.crosstab(
        eval_df["true_label"], 
        eval_df["pred_label"], 
        rownames=["True"], 
        colnames=["Pred"]
    )
    print(conf_matrix)
    print("=" * 65)

    out_csv = os.path.join(settings.BASE_DIR, "eval_results.csv")
    eval_df.to_csv(out_csv, index=False)
    print(f"\nDetailed evaluation results written to: {out_csv}\n")

if __name__ == "__main__":
    run_evaluation()
