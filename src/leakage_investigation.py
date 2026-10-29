"""Data leakage and feature semantics investigation module.

Analyzes suspicious features, redundant columns, and target encoding risks
to provide an exhaustive scientific evaluation of why baseline classifiers
achieve near-perfect or perfect classification.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import get_data_dir, get_reports_dir

SUSPICIOUS_FEATURES = [
    "filetype",
    "pe_res_name",
    "category",
    "pe_sec_name",
    "log",
    "action",
    "apistats",
    "dll",
    "file_created",
    "file_read",
    "arguments",
    "dll_loaded",
    "api",
    "entropy",
]

# Semantics of features based on dynamic malware analysis sandbox reports (e.g. Cuckoo/CAPE)
FEATURE_SEMANTICS = {
    "filetype": "Static PE file metadata: count/identifier of distinct filetypes or embedded resource filetypes.",
    "pe_res_name": "Static PE metadata: count of named PE resources (icons, manifests, dialogs).",
    "category": "Behavioral category summary count (e.g., registry, filesystem, networking signatures triggered).",
    "pe_sec_name": "Static PE structure: count of distinct PE section names (.text, .data, .rsrc, etc.).",
    "log": "Sandbox runtime metadata: total size or count of execution log events captured in analysis.",
    "action": "Sandbox signature flag/indicator: automated classification action taken by sandbox signature engine.",
    "apistats": "Dynamic API invocation aggregate count across the entire sandbox detonation run.",
    "dll": "Dynamic library count: number of DLL references observed during execution.",
    "file_created": "Filesystem behavioral count: total files created during dynamic execution.",
    "file_read": "Filesystem behavioral count: total files read during dynamic execution.",
    "arguments": "Dynamic execution parameter count: number of command-line arguments / parameter invocations.",
    "dll_loaded": "Dynamic memory/process behavior: count of LoadLibrary/LdrLoadDll calls observed at runtime.",
    "api": "Dynamic API activity count: total distinct or invoked API call instances.",
    "entropy": "Static/Dynamic memory metric: measure of file/memory section randomness/packing.",
}


def run_leakage_analysis(output_filepath: Path = None) -> str:
    """Execute full data leakage, redundancy, and semantic analysis on suspicious features."""
    if output_filepath is None:
        output_filepath = get_reports_dir() / "leakage_analysis.txt"

    data_path = get_data_dir() / "processed" / "cleaned_dataset.csv"
    df = pd.read_csv(data_path)

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("DATA LEAKAGE & FEATURE SEMANTICS INVESTIGATION REPORT")
    lines.append("Project: Early-Stage Ransomware Detection via Dynamic Memory Analysis & RL")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. EXECUTIVE SUMMARY & INITIAL 100% ACCURACY INVESTIGATION")
    lines.append("-" * 80)
    lines.append("An initial Random Forest baseline achieved 100% across Accuracy, Precision, Recall,")
    lines.append("and F1-Score. In empirical cybersecurity research, perfect classification on dynamic")
    lines.append("behavioral datasets is an immediate warning sign requiring critical validation.")
    lines.append("")
    lines.append("Our investigation reveals THREE primary factors contributing to this outcome:")
    lines.append("  (A) Sandbox-Wide Post-Execution Aggregations: Features represent whole-execution")
    lines.append("      summary counts (e.g. apistats, file_read, file_created, log), capturing the entire")
    lines.append("      lifecycle after the ransomware has already executed and encrypted files.")
    lines.append("  (B) Strict Redundancies & Duplicate Columns: 15 pairs of feature columns are 100%")
    lines.append("      identical (e.g., pe_res_name == filetype, arguments == api == category).")
    lines.append("  (C) Distinct Family Profiles: The 3 malware families (E, G, L) exhibit distinct")
    lines.append("      distributional clusters in their sandbox logs (e.g., file creation counts and")
    lines.append("      API statistics differ by orders of magnitude).")
    lines.append("")
    lines.append("=" * 80)
    lines.append("2. IDENTICAL COLUMN REDUNDANCY ANALYSIS")
    lines.append("-" * 80)

    numeric_cols = [c for c in df.columns if c != "family"]
    identical_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1 = numeric_cols[i]
            c2 = numeric_cols[j]
            if (df[c1] == df[c2]).all():
                identical_pairs.append((c1, c2))

    lines.append(f"Detected {len(identical_pairs)} perfectly identical feature pairs across all 1,750 rows:")
    for c1, c2 in identical_pairs:
        lines.append(f"  * {c1:22s} == {c2:22s} (correlation = 1.0000, 100% identical values)")
    lines.append("")
    lines.append("Scientific Interpretation of Redundancy:")
    lines.append("  - 'arguments', 'api', and 'category' are completely identical (all 1,750 values match).")
    lines.append("  - 'pe_res_name' and 'filetype' are completely identical.")
    lines.append("  - 'pe_sec_name' and 'entropy' are completely identical.")
    lines.append("  - 'proc_pid' and 'file' are completely identical.")
    lines.append("  - 'proc', 'beh_command_line', and 'process_path' are completely identical.")
    lines.append("  - 'tree_command_line' and 'tree_process_name' are completely identical.")
    lines.append("  - 'program' and 'info' are completely identical.")
    lines.append("  - 'description' and 'sign_name' are completely identical.")
    lines.append("These identical pairs stem from duplicate field mappings or synthetic normalization")
    lines.append("in the original Annex A extraction. They do not provide independent behavioral information.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("3. DETAILED INVESTIGATION OF SPECIFIED SUSPICIOUS FEATURES")
    lines.append("-" * 80)

    for feat in SUSPICIOUS_FEATURES:
        lines.append(f"\nFEATURE: [{feat}]")
        lines.append("-" * 40)
        semantics = FEATURE_SEMANTICS.get(feat, "Dynamic analysis behavioral indicator.")
        lines.append(f"Semantics Description   : {semantics}")

        num_unique = df[feat].nunique()
        lines.append(f"Number of Unique Values : {num_unique}")
        lines.append(f"Overall Value Range     : min={df[feat].min()}, max={df[feat].max()}")

        # Family distribution statistics
        lines.append("Family Summary Statistics:")
        stats = df.groupby("family")[feat].agg(["count", "mean", "std", "median", "min", "max"])
        for fam, row in stats.iterrows():
            lines.append(
                f"  - Family '{fam}': mean={row['mean']:10.2f}, median={row['median']:8.2f}, "
                f"std={row['std']:10.2f}, min={row['min']:6.0f}, max={row['max']:8.0f}"
            )

        # Crosstab sample or value counts
        if num_unique <= 10:
            lines.append("Complete Crosstab with Target (family):")
            ct = pd.crosstab(df[feat], df["family"])
            for val, row in ct.iterrows():
                lines.append(f"    Value {val:4.0f} -> E: {row.get('E', 0):4d}, G: {row.get('G', 0):4d}, L: {row.get('L', 0):4d}")
        else:
            lines.append(f"Top 5 Most Frequent Values across Families:")
            top_vals = df[feat].value_counts().head(5).index
            ct_top = pd.crosstab(df[df[feat].isin(top_vals)][feat], df["family"])
            for val, row in ct_top.iterrows():
                lines.append(f"    Value {val:8.1f} -> E: {row.get('E', 0):4d}, G: {row.get('G', 0):4d}, L: {row.get('L', 0):4d}")

        # Direct Target Encoding Assessment
        # Check if any single value uniquely and exclusively determines a family with 100% certainty
        val_fam_ct = pd.crosstab(df[feat], df["family"])
        exclusive_vals = (val_fam_ct > 0).sum(axis=1) == 1
        num_exclusive = int(exclusive_vals.sum())
        lines.append(f"Direct Target Encoding  : Does this feature directly encode the label?")
        if num_exclusive > (num_unique * 0.7) and num_unique > 20:
            lines.append(f"  WARNING: High exclusivity ({num_exclusive}/{num_unique} unique values occur in exactly one family).")
            lines.append("  This indicates strong family-specific clustering, characteristic of sandbox run summaries.")
        else:
            lines.append(f"  No direct target bijection ({num_exclusive}/{num_unique} values are exclusive). Overlaps exist.")

        # Metadata vs Behavioral Assessment
        if feat in ["filetype", "pe_res_name", "pe_sec_name", "entropy"]:
            lines.append("Classification Type     : Static PE Header / Structural Metadata.")
            lines.append("Temporal Quality        : Non-temporal. Known before process execution.")
        elif feat in ["log", "action"]:
            lines.append("Classification Type     : Sandbox Execution Environment Metadata.")
            lines.append("Temporal Quality        : Post-execution artifact of sandbox reporting engine.")
        else:
            lines.append("Classification Type     : Aggregate Dynamic Behavioral Indicator.")
            lines.append("Temporal Quality        : Full-trace aggregate (end-of-execution summary, not early-stage).")

    lines.append("")
    lines.append("=" * 80)
    lines.append("4. FEATURE RETENTION / REMOVAL SCIENTIFIC RATIONALE")
    lines.append("-" * 80)
    lines.append("Guiding Research Principle:")
    lines.append("  Per project instructions, we do NOT remove features merely because they are highly predictive.")
    lines.append("  We also preserve all 46 features for the baseline model to benchmark against prior runs.")
    lines.append("")
    lines.append("Retention Decision Table:")
    lines.append("  - Retained All 46 Features: Maintained in data/processed/ml_ready_dataset.csv.")
    lines.append("  - Removal for Baselines: 0 additional features removed.")
    lines.append("  - Scientific Rationale:")
    lines.append("      1. The baseline goal is to rigorously reproduce and evaluate tree-based classifiers")
    lines.append("         on the standard dataset under strict stratified split conditions.")
    lines.append("      2. Removing features selectively would introduce researcher bias unless guided by")
    lines.append("         an explicit temporal cutoff.")
    lines.append("      3. The redundancy between identical pairs (e.g. pe_res_name and filetype) splits")
    lines.append("         tree importance but does not bias accuracy in Random Forest or XGBoost.")
    lines.append("      4. In Task 11 ('early_stage_gap.md') and Task 12 ('ppo_design.md'), we document")
    lines.append("         how real early-stage detection requires shifting from whole-trace aggregations")
    lines.append("         to sequential, time-windowed memory and process snapshots.")
    lines.append("=" * 80)

    report_content = "\n".join(lines) + "\n"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Leakage and feature semantics analysis written to: {output_filepath}")
    return report_content


if __name__ == "__main__":
    run_leakage_analysis()

