#!/usr/bin/env python3
"""Compose the PNG previews of the paper figures.

The previews are NOT embedded in the manuscript.  This is a standalone
post-processing script:

    paper PDF panels  ->  figure/png/<semantic-name>.png

The input PDFs are the panels actually inserted into the paper.  The output
PNGs follow the paper layout: one row per subfigure row, subcaptions under each
panel, a separate legend row, and equal cell sizes inside each figure.

Usage (from the repository root):

    python tools/make_figure_previews.py

Optional:

    python tools/make_figure_previews.py --width 2400 --scale 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1]
PDF_ROOT = BASE / "figure" / "pdf"
PNG_ROOT = BASE / "figure" / "png"
MANIFEST = BASE / "figure" / "manifest.json"

MODELS = ["VGG-11", "Res-18", "ViT-8", "Mix-8"]
MODEL_NAMES = ["VGGNet", "ResNet", "ViT", "MLP-Mixer"]
CLASSES = ["air", "auto", "ship", "truck"]

# The axis titles in the paper panels use Liberation Serif at 32 pt.  The
# rendered preview is 2400 px wide, where this becomes 44 px; keep the two
# settings together if the output width is changed.
FONT_SIZE = 44
CAPTION_HEIGHT = 68


def _panel(path: str) -> str:
    return path if path.startswith("s-") else f"panels/{path}"


def _row(paths, captions=None):
    if captions is None:
        captions = [""] * len(paths)
    return [(_panel(p), c) for p, c in zip(paths, captions)]


def _overall(task_dir: str):
    return [
        _row([f"{task_dir}/VGG-11.pdf", f"{task_dir}/Res-18.pdf"], ["VGGNet", "ResNet"]),
        _row([f"{task_dir}/ViT-8.pdf", f"{task_dir}/Mix-8.pdf"], ["ViT", "MLP-Mixer"]),
        [(_panel("legend/conditions.pdf"), "")],
    ]


def _weight(fig_dir: str):
    return [
        _row(
            [f"{fig_dir}/VGG_L3.pdf", f"{fig_dir}/VGG_L5.pdf",
             f"{fig_dir}/VGG_L7.pdf", f"{fig_dir}/VGG_L9.pdf"],
            ["VGGNet 3-C", "VGGNet 5-C", "VGGNet 7-C", "VGGNet 9-D"],
        ),
        _row(
            [f"{fig_dir}/RES_L3.pdf", f"{fig_dir}/RES_L4.pdf",
             f"{fig_dir}/RES_L6.pdf", f"{fig_dir}/RES_L8.pdf"],
            ["ResNet 3-R", "ResNet 4-R", "ResNet 6-R", "ResNet 8-R"],
        ),
        [(_panel(f"{fig_dir}/legend.pdf"), "")],
    ]


def _classes():
    rows = [
        _row(
            [f"classification/class/c{i}/{m}.pdf" for m in MODELS],
            [f"{n} ({c})" for n in MODEL_NAMES],
        )
        for i, c in enumerate(CLASSES, 1)
    ]
    rows.append([(_panel("legend/conditions.pdf"), "")])
    return rows


def _thresholds(start: int, end: int, folder: str):
    rows = [
        _row(
            [f"regression/{folder}/t{t}/{m}.pdf" for m in MODELS],
            [f"{n} (τ{t})" for n in MODEL_NAMES],
        )
        for t in range(start, end + 1)
    ]
    rows.append([(_panel("legend/conditions.pdf"), "")])
    return rows


def _heatmap():
    rows = [
        _row(
            [f"heatmap/classification/{c}.pdf", f"heatmap/regression/{c}.pdf"],
            [name, name],
        )
        for c, name in (
            ("attack", "Attack"), ("gauss", "Gauss"), ("salt", "Salt"),
            ("move", "Move"), ("occ", "Occ"),
        )
    ]
    rows.append(_row(
        ["heatmap/cbar_classification.pdf", "heatmap/cbar_regression.pdf"],
        ["", ""],
    ))
    return rows


def _radar():
    return [
        _row([f"classification/radar/{m}.pdf" for m in MODELS], MODEL_NAMES),
        [(_panel("legend/training.pdf"), "")],
        _row(
            [f"regression/radar/{m}_gauss.pdf" for m in MODELS],
            [f"{n} (gauss)" for n in MODEL_NAMES],
        ),
        _row(
            [f"regression/radar/{m}_salt.pdf" for m in MODELS],
            [f"{n} (salt)" for n in MODEL_NAMES],
        ),
        [(_panel("legend/training.pdf"), "")],
    ]


def _metric(task: str):
    return [
        _row(
            [f"extra/noise_metric_{task}_gauss_{m}.pdf" for m in MODELS],
            [f"{n} (Gauss)" for n in MODEL_NAMES],
        ),
        _row(
            [f"extra/noise_metric_{task}_salt_{m}.pdf" for m in MODELS],
            [f"{n} (Salt)" for n in MODEL_NAMES],
        ),
        [(_panel("extra/legend_pair.pdf"), "")],
    ]


def _s_scan():
    return [
        _row(
            ["s_scan/s_scan_attack.pdf", "s_scan/s_scan_gauss.pdf",
             "s_scan/s_scan_salt.pdf"],
            ["Attack", "Gauss", "Salt"],
        ),
        _row(
            ["s_scan/s_scan_move.pdf", "s_scan/s_scan_occ.pdf"],
            ["Move", "Occ"],
        ),
        [(_panel("s_scan/legend_s_scan.pdf"), "")],
    ]


def build_layouts():
    """Return the paper figure list in the same order as the manuscript."""
    return [
        {
            "id": "sample_selection_noise",
            "label": "fig:sel_data",
            "caption": "Susceptible sample selection and noise imposition",
            "rows": [[("s-1.pdf", "Initial samples"), ("s-2.pdf", "perturbed samples")]],
        },
        {"id": "weight_similarity_sampling", "label": "fig:weight_vs_samples",
         "caption": "Weight cosine similarity under 10 random samplings",
         "rows": _weight("figure2")},
        {"id": "weight_stability_noise", "label": "fig:weight_vs_noise",
         "caption": "Pairwise cosine stability under 10 Gaussian-noise realizations",
         "rows": _weight("figure3")},
        {"id": "classification_overall", "label": "fig_noi_c_over",
         "caption": "Classification robustness criteria: overall",
         "rows": _overall("classification/overall")},
        {"id": "classification_per_class", "label": "fig_noi_c",
         "caption": "Classification robustness criteria: per-class OVR",
         "rows": _classes()},
        {"id": "classification_full_cifar10", "label": "fig:sub_cla_full",
         "caption": "Full CIFAR-10 robustness criteria",
         "rows": _overall("classification_full/overall")},
        {"id": "regression_overall", "label": "fig_noi_r_over",
         "caption": "Regression robustness criteria: overall",
         "rows": _overall("regression/overall")},
        {"id": "regression_thresholds_1_5", "label": "fig_noi_c1",
         "caption": "Regression robustness criteria: thresholds 1-5",
         "rows": _thresholds(1, 5, "thrA")},
        {"id": "regression_thresholds_6_9", "label": "fig_noi_c2",
         "caption": "Regression robustness criteria: thresholds 6-9",
         "rows": _thresholds(6, 9, "thrB")},
        {"id": "standard_error_heatmap", "label": "fig:std_heat",
         "caption": "Standard error heatmap",
         "rows": _heatmap(), "heatmap": True},
        {"id": "noise_trained_radar", "label": "fig:comp_acc",
         "caption": "Clean vs noise-trained comparison (radar)",
         "rows": _radar()},
        {"id": "classification_noise_metric", "label": "fig:comp_acc_s1",
         "caption": "Classification noise metric comparison",
         "rows": _metric("classification")},
        {"id": "regression_noise_metric", "label": "fig:comp_acc_s2",
         "caption": "Regression noise metric comparison",
         "rows": _metric("regression")},
        {"id": "susceptible_ratio_sensitivity", "label": "fig:s_scan",
         "caption": "Sensitivity to susceptible-sample ratio s",
         "rows": _s_scan()},
    ]


def _font():
    for path in (
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, FONT_SIZE)
    return ImageFont.load_default()


class Renderer:
    def __init__(self, scale: int):
        self.scale = scale
        self.cache = {}

    def image(self, relative_path: str) -> Image.Image:
        if relative_path not in self.cache:
            pdf = pdfium.PdfDocument(str(PDF_ROOT / relative_path))
            page = pdf[0]
            self.cache[relative_path] = page.render(scale=self.scale).to_pil().convert("RGB")
        return self.cache[relative_path]


def _pad_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width >= width:
        return image
    canvas = Image.new("RGB", (width, image.height), "white")
    canvas.paste(image, (width - image.width, 0))
    return canvas


def compose(fig, renderer: Renderer, out_path: Path, width: int):
    rows = [[(renderer.image(p), c) for p, c in row] for row in fig["rows"]]

    # The classification heatmap has y labels; the regression one does not.
    # Give the regression image the same left margin so both panels share one
    # outer width and height.
    if fig.get("heatmap"):
        fixed = []
        for row in rows:
            if len(row) == 2:
                (left, left_cap), (right, right_cap) = row
                right = _pad_to_width(right, left.width)
                fixed.append([(left, left_cap), (right, right_cap)])
            else:
                fixed.append(row)
        rows = fixed

    margin, gap = 40, 30
    panel_rows = [row for row in rows if not (len(row) == 1 and not row[0][1])]
    max_cols = max((len(row) for row in panel_rows), default=1)
    cell_w = (width - 2 * margin - (max_cols - 1) * gap) // max_cols
    image_h = max(
        int(image.height * cell_w / image.width)
        for row in panel_rows
        for image, _ in row
    )

    specs = []
    for row in rows:
        if len(row) == 1 and not row[0][1]:
            legend_w = int(width * 0.45)
            legend_h = int(row[0][0].height * legend_w / row[0][0].width)
            specs.append(("legend", legend_w, legend_h))
        else:
            specs.append(("grid", cell_w, image_h + (CAPTION_HEIGHT if any(c for _, c in row) else 0)))

    height = 2 * margin + sum(spec[2] for spec in specs) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    y = margin
    letter = 0

    for row, spec in zip(rows, specs):
        if spec[0] == "legend":
            legend_w, legend_h = spec[1], spec[2]
            image = row[0][0].resize((legend_w, legend_h), Image.Resampling.LANCZOS)
            canvas.paste(image, ((width - legend_w) // 2, y))
            y += legend_h + gap
            continue

        _, cell_w, cell_h = spec
        x = (width - (len(row) * cell_w + gap * (len(row) - 1))) // 2
        for image, caption in row:
            scale = min(cell_w / image.width, image_h / image.height)
            new_w = max(1, int(image.width * scale))
            new_h = max(1, int(image.height * scale))
            resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas.paste(
                resized,
                (x + (cell_w - new_w) // 2, y + (image_h - new_h) // 2),
            )
            if caption:
                draw.text(
                    (x + cell_w // 2, y + image_h + 8),
                    f"({chr(97 + letter)}) {caption}",
                    font=font,
                    fill="black",
                    anchor="ma",
                )
                letter += 1
            x += cell_w + gap
        y += cell_h + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300), optimize=True)


def main():
    global PDF_ROOT, PNG_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--pdf-root", type=Path, default=PDF_ROOT)
    parser.add_argument("--out", type=Path, default=PNG_ROOT)
    args = parser.parse_args()

    PDF_ROOT = args.pdf_root
    PNG_ROOT = args.out
    renderer = Renderer(args.scale)
    manifest = []

    for fig in build_layouts():
        out_path = PNG_ROOT / f"{fig['id']}.png"
        compose(fig, renderer, out_path, args.width)
        manifest.append({
            "id": fig["id"],
            "label": fig["label"],
            "caption": fig["caption"],
            "preview": str(out_path.relative_to(BASE)).replace("\\", "/"),
            "panels": fig["rows"],
        })
        print(f"[ok] {out_path}")

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] {MANIFEST}")


if __name__ == "__main__":
    main()

