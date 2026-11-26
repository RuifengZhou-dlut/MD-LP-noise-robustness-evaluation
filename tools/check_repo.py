#!/usr/bin/env python3
"""Static sanity check of the cleaned repository (no TensorFlow needed)."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADD_GROUPS = ["probe_efficiency", "probe_stability",
              "similarity_comparison", "s_sensitivity"]
ADD_SUBGROUPS = ["weight_vs_samples", "weight_vs_noise"]
errors, warnings = [], []


def need(path, kind="file"):
    p = ROOT / path
    ok = p.exists() and (p.is_file() if kind == "file" else p.is_dir())
    if not ok:
        errors.append(f"missing {kind}: {path}")


def main():
    for d in ["src", "model", "dataset", "result", "figure", "docs"]:
        need(d, "dir")

    for name in ["README.md", "requirements.txt", ".gitignore"]:
        need(name)

    need("src/add_exp/common", "dir")
    for add in ADD_GROUPS:
        need(f"src/add_exp/{add}", "dir")
        need(f"result/add_exp/{add}", "dir")
    for sub in ADD_SUBGROUPS:
        need(f"src/add_exp/probe_stability/{sub}", "dir")
        need(f"result/add_exp/probe_stability/{sub}", "dir")

    for name in ["architecture", "experiments", "data_and_results", "reproduction"]:
        need(f"docs/{name}.md")

    # Keep documentation centralised: no folder-level README files
    for area in ("src", "dataset", "model", "result", "figure"):
        for p in (ROOT / area).rglob("README*.md"):
            errors.append(f"folder-level README should be removed: {p.relative_to(ROOT)}")

    # Python syntax
    py = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").rglob("*.py"))
    for p in py:
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"syntax error {p.relative_to(ROOT)}: {exc}")

    # Code comments/docstrings must be English only
    cjk = re.compile(r"[\u4e00-\u9fff]")
    for p in (ROOT / "src").rglob("*.py"):
        if cjk.search(p.read_text(encoding="utf-8")):
            errors.append(f"non-English (CJK) text in {p.relative_to(ROOT)}")

    # Forbidden paths / stale modules
    for p in (ROOT / "src").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        if "/mnt/LSMN" in text:
            errors.append(f"absolute path in {p.relative_to(ROOT)}")
        for stale in ("code.common", "vggres.code", "src.analysis", "src.config"):
            if stale in text:
                warnings.append(f"stale reference '{stale}' in {p.relative_to(ROOT)}")

    # Assets
    models = list((ROOT / "model").rglob("*.h5"))
    if len(models) < 20:
        errors.append(f"expected 20 model weights, found {len(models)}")
    need("tools/make_figure_previews.py")
    need("figure/pdf/panels", "dir")
    need("figure/png", "dir")
    need("result/standard_error", "dir")
    need("result/repeat_raw", "dir")
    need("result/scoring", "dir")
    need("result/criteria/clean/classification_full/tidy", "dir")
    need("dataset/clean/classification/data.npz")
    need("dataset/clean/regression/parts", "dir")
    criteria = list((ROOT / "result/criteria/clean").glob("*.csv"))
    if len(criteria) < 2:
        errors.append("missing main criteria csv")

    # Markdown local links
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for md in [ROOT / "README.md"] + list((ROOT / "docs").glob("*.md")):
        for target in link_re.findall(md.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not (md.parent / target).exists():
                warnings.append(f"broken link in {md.relative_to(ROOT)}: {target}")

    # No committed bytecode
    for p in ROOT.rglob("__pycache__"):
        errors.append(f"committed __pycache__: {p.relative_to(ROOT)}")

    if warnings:
        print("WARNINGS")
        for w in warnings:
            print(" -", w)
    if errors:
        print("ERRORS")
        for e in errors:
            print(" -", e)
        return 1
    print(f"OK: {len(py)} python files, {len(models)} models, {len(criteria)} criteria csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
