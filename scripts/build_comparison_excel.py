import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import ColorScaleRule

ROOT = Path.cwd()
RESULTS = ROOT / "results"
PREDICTIONS = ROOT / "predictions"

INPUT_FILE = RESULTS / "evaluation_results.csv"
OUTPUT_FILE = RESULTS / "COLING_Detoxification_Complete_Comparison.xlsx"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing results file: {INPUT_FILE}")

# -------------------------------------------------------
# 1. Load consolidated results
# -------------------------------------------------------
results = pd.read_csv(INPUT_FILE)

required = ["model_name", "STA", "SIM", "ChrF1", "Joint",
            "num_examples", "failed_generations"]

missing = [c for c in required if c not in results.columns]
if missing:
    raise ValueError(f"Missing columns in evaluation results: {missing}")

results = results.drop_duplicates("model_name", keep="last")
results = results.sort_values("Joint", ascending=False).reset_index(drop=True)

# -------------------------------------------------------
# 2. Build model comparison sheet
# -------------------------------------------------------
comparison = results.copy()
comparison.insert(0, "Joint Rank", range(1, len(comparison) + 1))

comparison["Evaluation Status"] = comparison.apply(
    lambda r: "PASS" if r["num_examples"] == 600
    and r["failed_generations"] == 0 else "CHECK",
    axis=1
)

# -------------------------------------------------------
# 3. Original paper comparison template
# -------------------------------------------------------
# IMPORTANT:
# Enter only values reported in the original paper, and only
# compare metrics when their definitions are compatible.
paper_comparison = results[
    ["model_name", "STA", "SIM", "ChrF1", "Joint"]
].copy()

paper_comparison = paper_comparison.rename(columns={
    "STA": "Our STA",
    "SIM": "Our SIM",
    "ChrF1": "Our ChrF1",
    "Joint": "Our Joint"
})

paper_comparison["Original Paper STA"] = pd.NA
paper_comparison["Original Paper SIM"] = pd.NA
paper_comparison["Original Paper ChrF1"] = pd.NA
paper_comparison["Original Paper Joint"] = pd.NA

paper_comparison["STA Difference"] = pd.NA
paper_comparison["SIM Difference"] = pd.NA
paper_comparison["ChrF1 Difference"] = pd.NA
paper_comparison["Joint Difference"] = pd.NA

paper_comparison["Comparison Notes"] = (
    "Enter the paper's reported score only if the model and metric "
    "definition match. Blank values mean not yet verified."
)

# -------------------------------------------------------
# 4. Per-example metrics from all evaluated models
# -------------------------------------------------------
per_example_frames = []

for path in sorted(RESULTS.glob("*_per_example_metrics.csv")):
    try:
        df = pd.read_csv(path)
        df.insert(0, "source_file", path.name)
        per_example_frames.append(df)
    except Exception as exc:
        print(f"Warning: could not read {path.name}: {exc}")

if per_example_frames:
    per_example = pd.concat(per_example_frames, ignore_index=True, sort=False)
else:
    per_example = pd.DataFrame(
        {"message": ["No per-example metric files found."]}
    )

# -------------------------------------------------------
# 5. Prediction file inventory
# -------------------------------------------------------
inventory = []

for folder in [PREDICTIONS, RESULTS]:
    if folder.exists():
        for path in sorted(folder.glob("*.csv")):
            try:
                df = pd.read_csv(path)
                inventory.append({
                    "Folder": str(path.parent.relative_to(ROOT)),
                    "Filename": path.name,
                    "Rows": len(df),
                    "Columns": ", ".join(df.columns.astype(str)),
                    "File Size (KB)": round(path.stat().st_size / 1024, 2)
                })
            except Exception as exc:
                inventory.append({
                    "Folder": str(path.parent.relative_to(ROOT)),
                    "Filename": path.name,
                    "Rows": "ERROR",
                    "Columns": str(exc),
                    "File Size (KB)": ""
                })

inventory_df = pd.DataFrame(inventory)

# -------------------------------------------------------
# 6. Write everything into ONE Excel workbook
# -------------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    comparison.to_excel(writer, sheet_name="Model Comparison", index=False)
    paper_comparison.to_excel(
        writer, sheet_name="Original vs Reproduced", index=False
    )
    results.to_excel(writer, sheet_name="Raw Results", index=False)
    per_example.to_excel(writer, sheet_name="Per-Example Metrics", index=False)
    inventory_df.to_excel(writer, sheet_name="File Inventory", index=False)

# -------------------------------------------------------
# 7. Format workbook
# -------------------------------------------------------
wb = load_workbook(OUTPUT_FILE)

for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False

    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )

    ws.row_dimensions[1].height = 30

    for column_cells in ws.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter

        for cell in column_cells[:2000]:
            value = str(cell.value) if cell.value is not None else ""
            max_length = max(max_length, min(len(value), 55))

        ws.column_dimensions[column_letter].width = min(
            max(max_length + 2, 12), 45
        )

# Color-scale the scores in the comparison sheet.
ws = wb["Model Comparison"]
headers = {
    cell.value: cell.column
    for cell in ws[1]
}

for metric in ["STA", "SIM", "ChrF1", "Joint"]:
    col = headers.get(metric)
    if col and ws.max_row >= 2:
        letter = ws.cell(1, col).column_letter
        ws.conditional_formatting.add(
            f"{letter}2:{letter}{ws.max_row}",
            ColorScaleRule(
                start_type="min", start_color="F8696B",
                mid_type="percentile", mid_value=50, mid_color="FFEB84",
                end_type="max", end_color="63BE7B"
            )
        )

# Add formulas so differences calculate automatically when paper
# scores are entered into the Original vs Reproduced sheet.
ws = wb["Original vs Reproduced"]
header_cols = {
    cell.value: cell.column
    for cell in ws[1]
}

for row in range(2, ws.max_row + 1):
    for metric in ["STA", "SIM", "ChrF1", "Joint"]:
        original_col = header_cols[f"Original Paper {metric}"]
        our_col = header_cols[f"Our {metric}"]
        diff_col = header_cols[f"{metric} Difference"]

        original_letter = ws.cell(1, original_col).column_letter
        our_letter = ws.cell(1, our_col).column_letter
        diff_letter = ws.cell(1, diff_col).column_letter

        ws.cell(row, diff_col).value = (
            f'=IF({original_letter}{row}="","",'
            f'{our_letter}{row}-{original_letter}{row})'
        )

wb.save(OUTPUT_FILE)

print("\n==========================================")
print("EXCEL WORKBOOK CREATED SUCCESSFULLY")
print("==========================================")
print(f"Output: {OUTPUT_FILE}")
print(f"Models: {len(results)}")
print(f"Per-example metric rows: {len(per_example)}")
print("\nModel ranking by Joint:")
print(results[["model_name", "Joint"]].to_string(index=False))
print("\nSheets:")
for sheet in wb.sheetnames:
    print(" -", sheet)
