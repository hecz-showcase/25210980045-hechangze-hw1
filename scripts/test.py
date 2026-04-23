"""Test entrypoint: evaluate best checkpoint on independent test split."""

import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PREP, PATHS, TEST, TRAIN
from src.eval.tester import Tester, load_class_names, load_test_split

# Error-analysis settings: pick top-n most frequent confusion pairs a->b,
# then show m wrong examples for each selected pair.
N_TOP_ERROR_PAIRS = 4
M_ERROR_CASES_PER_PAIR = 4


def _format_confusion_matrix_lines(cm: np.ndarray, class_names: List[str]) -> List[str]:
    lines: List[str] = []
    lines.append("Confusion Matrix (rows=true, cols=pred):")
    if class_names and len(class_names) == cm.shape[0]:
        header = " " * 18 + " ".join([f"{name[:12]:>12}" for name in class_names])
        lines.append(header)
        for i, row in enumerate(cm):
            row_name = class_names[i][:16]
            row_values = " ".join([f"{int(v):>12}" for v in row])
            lines.append(f"{row_name:>16} {row_values}")
    else:
        lines.extend(str(cm).splitlines())
    return lines


def _print_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    for line in _format_confusion_matrix_lines(cm, class_names):
        print(line)


def _save_test_outputs(
    acc: float,
    cm: np.ndarray,
    class_names: List[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_path = output_dir / "test_results.json"
    payload = {
        "accuracy": acc,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable text summary
    txt_path = output_dir / "test_results.txt"
    lines = [f"Test Accuracy: {acc:.6f}"]
    lines.extend(_format_confusion_matrix_lines(cm, class_names))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CSV matrix
    csv_path = output_dir / "test_confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        if class_names and len(class_names) == cm.shape[0]:
            f.write("," + ",".join(class_names) + "\n")
            for i, row in enumerate(cm):
                f.write(class_names[i] + "," + ",".join(str(int(v)) for v in row) + "\n")
        else:
            for row in cm:
                f.write(",".join(str(int(v)) for v in row) + "\n")

    print(f"Saved test JSON: {json_path}")
    print(f"Saved test TXT:  {txt_path}")
    print(f"Saved test CSV:  {csv_path}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _to_display_image(x_flat: np.ndarray, image_size: int) -> np.ndarray:
    img = x_flat.reshape(image_size, image_size, 3)
    if np.max(img) > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _run_error_analysis(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_fig_dir: Path,
    n_top_pairs: int,
    m_cases_per_pair: int,
) -> None:
    num_classes = probs.shape[1]
    m_cases_per_pair = max(1, m_cases_per_pair)

    pair_counts: List[tuple[int, int, int]] = []
    for true_c in range(num_classes):
        for pred_c in range(num_classes):
            if true_c == pred_c:
                continue
            count = int(np.sum((y_test == true_c) & (y_pred == pred_c)))
            if count > 0:
                pair_counts.append((true_c, pred_c, count))

    pair_counts.sort(key=lambda t: t[2], reverse=True)
    selected = pair_counts[: max(1, min(n_top_pairs, len(pair_counts)))]

    print("Error analysis (top confusion pairs a->b):")
    for true_c, pred_c, count in selected:
        true_name = class_names[true_c] if true_c < len(class_names) else f"class_{true_c}"
        pred_name = class_names[pred_c] if pred_c < len(class_names) else f"class_{pred_c}"
        print(f"  {true_name} -> {pred_name}: {count}")

    fig, axes = plt.subplots(
        nrows=len(selected),
        ncols=m_cases_per_pair,
        figsize=(3.2 * m_cases_per_pair, 3.2 * len(selected)),
    )
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)
    if m_cases_per_pair == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (true_c, pred_c, count) in enumerate(selected):
        true_name = class_names[true_c] if true_c < len(class_names) else f"class_{true_c}"
        pred_name = class_names[pred_c] if pred_c < len(class_names) else f"class_{pred_c}"
        pair_idx = np.where((y_test == true_c) & (y_pred == pred_c))[0]

        if pair_idx.size > 0:
            # Rank a->b cases by confidence on b.
            wrong_conf = probs[pair_idx, pred_c]
            order = np.argsort(wrong_conf)[::-1]
            chosen = pair_idx[order[:m_cases_per_pair]]
        else:
            chosen = np.array([], dtype=np.int64)

        for col_idx in range(m_cases_per_pair):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= chosen.size:
                if col_idx == 0:
                    ax.set_title(f"{true_name}->{pred_name}\ncount={count}\n(no case)", fontsize=9)
                continue

            idx = int(chosen[col_idx])
            sample_true = int(y_test[idx])
            sample_pred = int(y_pred[idx])
            pred_prob = float(probs[idx, sample_pred])

            img = _to_display_image(x_test[idx], DATA_PREP.image_size)
            ax.imshow(img)
            t_name = class_names[sample_true] if sample_true < len(class_names) else f"class_{sample_true}"
            p_name = class_names[sample_pred] if sample_pred < len(class_names) else f"class_{sample_pred}"
            ax.set_title(
                f"true={t_name}\npred={p_name}\nconf={pred_prob:.3f}",
                fontsize=8,
            )

    plt.tight_layout()
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_fig_dir / f"test_error_pair_matrix_n{len(selected)}_m{m_cases_per_pair}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"Saved error-case matrix: {fig_path}")


def main() -> None:
    checkpoint_path = PATHS.checkpoints_dir / TEST.checkpoint_name
    split_path = PATHS.outputs_dir / f"{DATA_PREP.file_prefix}.npz"
    meta_path = PATHS.outputs_dir / f"{DATA_PREP.file_prefix}_meta.json"

    x_test, y_test = load_test_split(split_path)

    tester = Tester(checkpoint_path=checkpoint_path, activation=TRAIN.activation)
    result = tester.evaluate(x_test=x_test, y_test=y_test)

    acc = float(result["accuracy"])
    cm = result["confusion_matrix"]
    y_pred = result["y_pred"]
    class_names = load_class_names(meta_path)

    print(f"Test Accuracy: {acc:.6f}")
    _print_confusion_matrix(cm, class_names)

    _save_test_outputs(
        acc=acc,
        cm=cm,
        class_names=class_names,
        output_dir=PATHS.outputs_dir / "logs",
    )

    # Additional error analysis: n*m matrix of high-confidence a->b confusion cases.
    if tester.model is None:
        tester.load_checkpoint()
    assert tester.model is not None
    logits = tester.model.forward(x_test)
    probs = _softmax(logits)
    _run_error_analysis(
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        probs=probs,
        class_names=class_names,
        output_fig_dir=PATHS.outputs_dir / "figures",
        n_top_pairs=N_TOP_ERROR_PAIRS,
        m_cases_per_pair=M_ERROR_CASES_PER_PAIR,
    )


if __name__ == "__main__":
    main()
