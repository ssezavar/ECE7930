#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Create a 1-page PDF report summarizing metrics and plots.

import argparse, os, math
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

def safe_read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None

def num(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return "-"

def build_table(df: pd.DataFrame):
    cols = ["dataset","accuracy","precision","recall","f1","auroc"]
    data = [ [c.upper() for c in cols] ]
    for _, r in df.iterrows():
        data.append([r.get("dataset","-")] + [num(r.get(c)) for c in cols[1:]])
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eeeeee")),
        ("TEXTCOLOR",(0,0),(-1,0), colors.black),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),0.3, colors.HexColor("#999999")),
        ("BOTTOMPADDING",(0,0),(-1,0),6),
        ("ALIGN",(1,1),(-1,-1),"CENTER")
    ]))
    return t

def add_image_if_exists(story, path: Path, width=460):
    if path.exists():
        img = Image(str(path), width=width, height=width*0.62)  # auto-ish aspect
        img.hAlign = "LEFT"
        story.append(img)
        story.append(Spacer(1,10))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaldir", default="ppd_eval_pro")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--title", default="PPD Synthetic Data Evaluation")
    args = ap.parse_args()

    evaldir = Path(args.evaldir)
    outdir = Path(args.outdir) if args.outdir else evaldir
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "ppd_report.pdf"

    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(args.title, styles["Title"]))
    story.append(Spacer(1,8))

    # Overall metrics
    overall = safe_read_csv(evaldir / "cls_metrics_overall.csv")
    if overall is not None:
        story.append(Paragraph("Overall classification (real test set):", styles["Heading3"]))
        story.append(build_table(overall))
        story.append(Spacer(1,10))

    # Fidelity
    fidelity = safe_read_csv(evaldir / "fidelity_centroid_cosine.csv")
    if fidelity is not None and "value" in fidelity.columns:
        story.append(Paragraph(f"Fidelity (centroid cosine, TF-IDF+SVD): <b>{num(fidelity['value'].iloc[0],3)}</b>", styles["Normal"]))
        story.append(Spacer(1,8))

    # Plots
    story.append(Paragraph("Figures:", styles["Heading3"]))
    add_image_if_exists(story, evaldir / "roc_curves.png")
    add_image_if_exists(story, evaldir / "bar_accuracy.png")
    add_image_if_exists(story, evaldir / "bar_f1.png")
    add_image_if_exists(story, evaldir / "privacy_hist.png")

    # Stratified metrics note
    bys = evaldir / "cls_metrics_by_stratum.csv"
    if bys.exists():
        story.append(Paragraph("Stratified metrics file available: cls_metrics_by_stratum.csv", styles["Italic"]))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    print("Saved report:", pdf_path)

if __name__ == "__main__":
    main()
