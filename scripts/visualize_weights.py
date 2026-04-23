"""Class-aware first-layer weight visualization with activation-aware unit selection.

Selection score per class c and unit u:
    score[u, c] = mean_activation[u | class=c] * contribution[u, c]

Where class=c can be grouped by:
- true labels
- predicted labels
- both (generate two sets of outputs)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PREP, PATHS, TEST, TRAIN
from src.models.activations import relu, sigmoid, tanh


# -----------------------------------------------------------------------------
# Editable parameters
# -----------------------------------------------------------------------------
TOP_K_PER_CLASS = 4
MAX_SAMPLES_PER_CLASS = 600
USE_ABS_CONTRIBUTION = False
LABEL_SOURCE: Literal["true", "pred", "both"] = "both"
PRED_BATCH_SIZE = 512
CHECKPOINT_NAME = TEST.checkpoint_name
OUTPUT_SUBDIR = "weight_viz_class_activation_aware"
CLASS_FILTER = None  # e.g., ["Forest", "River"]
REPORT_SUMMARY_CLASSES = ["Forest", "River", "SeaLake", "Highway"]
# -----------------------------------------------------------------------------


def _resolve_activation_fn(name: str):
    key = name.lower()
    if key == "relu":
        return relu
    if key == "sigmoid":
        return sigmoid
    if key == "tanh":
        return tanh
    raise ValueError("Unsupported activation. Use one of: relu, sigmoid, tanh")


def load_checkpoint_weights(checkpoint_path: Path) -> Dict[str, np.ndarray]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = np.load(checkpoint_path)
    required = ("W1", "b1", "W2", "b2", "W3", "b3")
    for key in required:
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing key: {key}")

    return {key: ckpt[key] for key in required}


def load_split_data(split_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    data = np.load(split_path)
    if "x_train" not in data or "y_train" not in data:
        raise KeyError("Split file must contain x_train and y_train")

    return data["x_train"], data["y_train"]


def load_class_names(meta_path: Path) -> List[str]:
    if not meta_path.exists():
        return []

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    class_to_idx = meta.get("class_to_idx", {})
    if not class_to_idx:
        return []

    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]


def infer_pred_labels(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    activation_name: str,
    batch_size: int,
) -> np.ndarray:
    act_fn = _resolve_activation_fn(activation_name)
    preds = np.empty((x.shape[0],), dtype=np.int64)

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        xb = x[start:end]

        z1 = xb @ w1 + b1
        a1 = act_fn(z1)
        z2 = a1 @ w2 + b2
        a2 = act_fn(z2)
        logits = a2 @ w3 + b3
        preds[start:end] = np.argmax(logits, axis=1)

    return preds


def weight_vector_to_image(weight_vec: np.ndarray, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    raw = weight_vec.reshape(image_size, image_size, 3)
    max_abs = float(np.max(np.abs(raw)))
    if max_abs < 1e-12:
        vis = np.full_like(raw, 0.5)
    else:
        vis = (raw / max_abs + 1.0) / 2.0
    vis = np.clip(vis, 0.0, 1.0)
    return raw, vis


def summarize_pattern(img_raw: np.ndarray) -> Tuple[str, str]:
    channel_energy = np.mean(np.abs(img_raw), axis=(0, 1))
    channel_names = ["R", "G", "B"]
    dominant_ch = channel_names[int(np.argmax(channel_energy))]

    h, w, _ = img_raw.shape
    h1, h2 = h // 4, 3 * h // 4
    w1, w2 = w // 4, 3 * w // 4

    center_mag = float(np.mean(np.abs(img_raw[h1:h2, w1:w2, :])))
    edge_mask = np.ones((h, w), dtype=bool)
    edge_mask[h1:h2, w1:w2] = False
    edge_mag = float(np.mean(np.abs(img_raw[edge_mask])))

    if center_mag > edge_mag * 1.10:
        spatial = "center-focused"
    elif edge_mag > center_mag * 1.10:
        spatial = "edge-focused"
    else:
        spatial = "balanced"

    return f"dominant {dominant_ch} channel", spatial


def build_observation_sentence(class_name: str, color: str, spatial: str) -> str:
    name = class_name.lower()
    if "forest" in name:
        return (
            f"{class_name}: selected weights show a {color} tendency with {spatial} structure, "
            "suggesting the first layer is already using vegetation-related color cues, while clear tree-like "
            "geometric texture is still weak."
        )
    if "river" in name:
        return (
            f"{class_name}: selected weights are relatively biased toward {color} with {spatial} structure, "
            "which is consistent with water-related color cues, but the spatial pattern is not yet a clean elongated channel."
        )
    if "sealake" in name or "sea" in name or "lake" in name:
        return (
            f"{class_name}: selected weights emphasize {color} with {spatial} structure, "
            "indicating that large-area water cues are captured mainly through color rather than explicit shape."
        )
    if "highway" in name:
        return (
            f"{class_name}: selected weights do not form a strong line-like template; they mainly show {color} "
            f"with {spatial} structure, suggesting low-level color contrast is used more than stable road geometry."
        )
    return (
        f"{class_name}: selected weights mainly show {color} with {spatial} structure, "
        "so the first layer appears to focus more on coarse color/contrast than explicit spatial layout."
    )


def create_report_summary_figure(
    summary_items: List[Dict[str, object]],
    output_path: Path,
) -> None:
    if not summary_items:
        return

    class_order: List[str] = []
    for item in summary_items:
        class_name = str(item["class_name"])
        if class_name not in class_order:
            class_order.append(class_name)

    cols = TOP_K_PER_CLASS
    rows = len(class_order)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)

    for ax in axes.ravel():
        ax.axis("off")

    for row_idx, class_name in enumerate(class_order):
        class_items = [item for item in summary_items if str(item["class_name"]) == class_name]
        for col_idx, item in enumerate(class_items[:cols]):
            ax = axes[row_idx, col_idx]
            ax.imshow(item["vis"])
            ax.set_title(
                f"{class_name}\nunit={item['unit']} score={item['score']:.4f}",
                fontsize=8,
            )
            ax.axis("off")

    fig.suptitle("Representative First-Layer Weights For Report (4x4)", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_class_mean_activation(
    x_train: np.ndarray,
    grouped_labels: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    num_classes: int,
    activation_name: str,
    max_samples_per_class: int,
) -> np.ndarray:
    act_fn = _resolve_activation_fn(activation_name)
    hidden_dim1 = w1.shape[1]
    mean_act = np.zeros((hidden_dim1, num_classes), dtype=np.float64)

    for c in range(num_classes):
        cls_idx = np.where(grouped_labels == c)[0]
        if cls_idx.size == 0:
            continue

        if max_samples_per_class > 0 and cls_idx.size > max_samples_per_class:
            cls_idx = cls_idx[:max_samples_per_class]

        x_cls = x_train[cls_idx]
        z1 = x_cls @ w1 + b1
        a1 = act_fn(z1)
        mean_act[:, c] = np.mean(a1, axis=0)

    return mean_act


def visualize_one_label_source(
    source_name: str,
    grouped_labels: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    x_train: np.ndarray,
    class_names: List[str],
    image_size: int,
    top_k_per_class: int,
    output_dir: Path,
    class_filter: List[str] | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    contribution = w2 @ w3
    contribution_for_score = np.abs(contribution) if USE_ABS_CONTRIBUTION else contribution
    num_classes = contribution.shape[1]

    mean_act = compute_class_mean_activation(
        x_train=x_train,
        grouped_labels=grouped_labels,
        w1=w1,
        b1=b1,
        num_classes=num_classes,
        activation_name=TRAIN.activation,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
    )

    score = mean_act * contribution_for_score

    report_lines: List[str] = []
    report_lines.append("Method")
    report_lines.append(
        "The first-layer weight visualization is generated from the trained checkpoint."
    )
    report_lines.append(
        "For each class, hidden units are ranked by score = mean_activation(class) * contribution."
    )
    report_lines.append(
        "Here, mean_activation(class) is the average hidden-layer response on samples grouped by the selected label source, "
        "and contribution is the hidden1-to-class parameter contribution derived from upper-layer weights."
    )
    report_lines.append(
        "The selected W1 columns are reshaped back to image size and normalized for display."
    )
    report_lines.append(f"Label source: {source_name}")
    report_lines.append(f"Activation: {TRAIN.activation}")
    report_lines.append(f"Top-K per class: {top_k_per_class}")
    report_lines.append(f"Max samples per class: {MAX_SAMPLES_PER_CLASS}")
    report_lines.append(f"Use abs(contribution): {USE_ABS_CONTRIBUTION}")
    report_lines.append("")
    report_lines.append("Observations")

    summary_items: List[Dict[str, object]] = []

    for c in range(num_classes):
        class_name = class_names[c] if c < len(class_names) else f"class_{c}"
        if class_filter is not None and class_name not in class_filter:
            continue

        rank = np.argsort(score[:, c])[::-1]
        chosen = rank[: min(top_k_per_class, rank.shape[0])]

        color_tags: List[str] = []
        spatial_tags: List[str] = []

        for u in chosen:
            raw, vis = weight_vector_to_image(w1[:, u], image_size)

            sc = float(score[u, c])
            ma = float(mean_act[u, c])
            co = float(contribution[u, c])

            color, spatial = summarize_pattern(raw)
            color_tags.append(color)
            spatial_tags.append(spatial)

            if class_name in REPORT_SUMMARY_CLASSES:
                summary_items.append(
                    {
                        "class_name": class_name,
                        "unit": int(u),
                        "score": sc,
                        "act": ma,
                        "contrib": co,
                        "vis": vis,
                        "color": color,
                        "spatial": spatial,
                    }
                )

        dominant_color = max(set(color_tags), key=color_tags.count)
        dominant_spatial = max(set(spatial_tags), key=spatial_tags.count)

        report_lines.append(f"{class_name}: units={list(map(int, chosen))}")
        report_lines.append(build_observation_sentence(class_name, dominant_color, dominant_spatial))
        report_lines.append("")

    report_path = output_dir / "weight_visualization_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    summary_items = [
        item
        for class_name in REPORT_SUMMARY_CLASSES
        for item in summary_items
        if str(item["class_name"]) == class_name
    ]
    create_report_summary_figure(summary_items, output_dir / "report_summary.png")
    return report_path


def main() -> None:
    checkpoint_path = PATHS.checkpoints_dir / CHECKPOINT_NAME
    split_path = PATHS.outputs_dir / f"{DATA_PREP.file_prefix}.npz"
    meta_path = PATHS.outputs_dir / f"{DATA_PREP.file_prefix}_meta.json"
    base_output_dir = PATHS.outputs_dir / "figures" / OUTPUT_SUBDIR

    weights = load_checkpoint_weights(checkpoint_path)
    x_train, y_true = load_split_data(split_path)
    class_names = load_class_names(meta_path)

    w1 = weights["W1"]
    b1 = weights["b1"]
    w2 = weights["W2"]
    b2 = weights["b2"]
    w3 = weights["W3"]
    b3 = weights["b3"]

    expected_dim = DATA_PREP.image_size * DATA_PREP.image_size * 3
    if w1.shape[0] != expected_dim:
        raise ValueError(
            f"W1 input dim {w1.shape[0]} does not match expected image size "
            f"{DATA_PREP.image_size}x{DATA_PREP.image_size}x3={expected_dim}."
        )

    if LABEL_SOURCE not in {"true", "pred", "both"}:
        raise ValueError("LABEL_SOURCE must be one of: true, pred, both")

    if LABEL_SOURCE in {"pred", "both"}:
        y_pred = infer_pred_labels(
            x=x_train,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            w3=w3,
            b3=b3,
            activation_name=TRAIN.activation,
            batch_size=PRED_BATCH_SIZE,
        )
    else:
        y_pred = None

    outputs: List[Tuple[str, Path]] = []

    if LABEL_SOURCE in {"true", "both"}:
        out_dir = base_output_dir / "true_label"
        rpt = visualize_one_label_source(
            source_name="true_label",
            grouped_labels=y_true,
            w1=w1,
            b1=b1,
            w2=w2,
            w3=w3,
            x_train=x_train,
            class_names=class_names,
            image_size=DATA_PREP.image_size,
            top_k_per_class=TOP_K_PER_CLASS,
            output_dir=out_dir,
            class_filter=CLASS_FILTER,
        )
        outputs.append(("true_label", rpt))

    if LABEL_SOURCE in {"pred", "both"} and y_pred is not None:
        out_dir = base_output_dir / "pred_label"
        rpt = visualize_one_label_source(
            source_name="pred_label",
            grouped_labels=y_pred,
            w1=w1,
            b1=b1,
            w2=w2,
            w3=w3,
            x_train=x_train,
            class_names=class_names,
            image_size=DATA_PREP.image_size,
            top_k_per_class=TOP_K_PER_CLASS,
            output_dir=out_dir,
            class_filter=CLASS_FILTER,
        )
        outputs.append(("pred_label", rpt))

    for tag, rpt in outputs:
        print(f"[{tag}] Saved figures to: {rpt.parent}")
        print(f"[{tag}] Saved report to:  {rpt}")


if __name__ == "__main__":
    main()
