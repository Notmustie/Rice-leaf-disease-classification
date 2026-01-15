import os, csv
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image

from src.xai.io import ensure_dir, collect_one_level
from src.xai.gradcam import preprocess_image_array, gradcam_from_feature_layer, overlay_heatmap
from src.xai.metrics import focus_and_entropy


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_preprocess(img_path: str, img_size, preprocess: str):
    img = Image.open(img_path).convert("RGB").resize(tuple(img_size))
    x = np.array(img).astype(np.float32)
    x = preprocess_image_array(x, preprocess)
    x = np.expand_dims(x, axis=0)
    return img, x

def topk(probs: np.ndarray, class_names, k: int):
    idx = probs.argsort()[::-1][:k]
    return [(class_names[i], float(probs[i]), int(i)) for i in idx]

def main(cfg_path: str = "configs/xai.yaml"):
    cfg = load_cfg(cfg_path)

    model = tf.keras.models.load_model(cfg["model_path"], compile=False)
    model_name = cfg.get("model_name", "model")

    out_root = os.path.join("outputs", "xai", model_name)
    overlays_dir = os.path.join(out_root, "overlays")
    heatmaps_dir = os.path.join(out_root, "heatmaps")
    ensure_dir(out_root); ensure_dir(overlays_dir); ensure_dir(heatmaps_dir)

    samples = collect_one_level(cfg["predict_root"], cfg.get("max_per_class", None))
    class_names = cfg["class_names"]
    img_size = tuple(cfg["img_size"])
    preprocess = cfg["preprocess"]
    top_k = int(cfg.get("top_k", 3))
    thr = float(cfg.get("reject_threshold", 0.60))
    feature_layer = cfg["feature_layer"]

    csv_path = os.path.join(out_root, "xai_summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_path","true_label",
            "pred_label","pred_conf","is_correct","rejected",
            "topk","feature_layer",
            "overlay_path","heatmap_path",
            "focus_score_ge_0.60","entropy"
        ])

        for item in samples:
            img_path = item["path"]
            true_label = item["true_label"]

            try:
                pil_img, x_in = load_and_preprocess(img_path, img_size, preprocess)
            except Exception as e:
                print("[skip load]", img_path, e)
                continue

            probs = model.predict(x_in, verbose=0)[0]
            tk = topk(probs, class_names, top_k)
            pred_label, pred_conf, pred_idx = tk[0]

            is_correct = (pred_label == true_label)
            rejected = (pred_conf < thr)

            try:
                heatmap = gradcam_from_feature_layer(
                    model=model,
                    x_in=x_in,
                    feature_layer_name=feature_layer,
                    class_index=pred_idx,
                    out_size=img_size
                )
            except Exception as e:
                print("[skip gradcam]", os.path.basename(img_path), e)
                continue

            overlay = overlay_heatmap(pil_img, heatmap, alpha=0.45)
            focus, ent = focus_and_entropy(heatmap, thr=0.60)

            base = os.path.splitext(os.path.basename(img_path))[0]
            overlay_path = os.path.join(overlays_dir, f"{base}_overlay.png")
            heatmap_path = os.path.join(heatmaps_dir, f"{base}_heatmap.png")

            overlay.save(overlay_path)
            Image.fromarray(np.uint8(255 * heatmap)).save(heatmap_path)

            topk_str = "; ".join([f"{n}:{c:.4f}" for n,c,_ in tk])

            w.writerow([
                img_path, true_label,
                pred_label, f"{pred_conf:.6f}", str(is_correct), str(rejected),
                topk_str, feature_layer,
                overlay_path, heatmap_path,
                f"{focus:.6f}", f"{ent:.6f}"
            ])

            status = "REJECTED" if rejected else "OK"
            corr = "CORRECT" if is_correct else "MISCLASS"
            print(f"[{status} | {corr}] {os.path.basename(img_path)} -> {pred_label} ({pred_conf:.3f})")

    # optional: aggregate table for report
    import pandas as pd
    df = pd.read_csv(csv_path)
    agg = df.groupby(["rejected"]).agg(
        samples=("image_path","count"),
        mean_conf=("pred_conf","mean"),
        mean_focus=("focus_score_ge_0.60","mean"),
        mean_entropy=("entropy","mean")
    ).reset_index()
    agg_path = os.path.join(out_root, "xai_agg.csv")
    agg.to_csv(agg_path, index=False)

    print("\nDone.")
    print("Summary:", csv_path)
    print("Aggregate:", agg_path)
    print("Outputs:", out_root)

if __name__ == "__main__":
    main()
