# day7_review.py
#
# This is NOT a training script. My plan is explicit that Day 7 has no
# new model to run -- it's the audit day: check everything Days 1-6
# produced is consistent, build one master results table out of the six
# separate result CSVs, and flag anything that needs a second look
# before Week 2 (writing) starts. No torch, no yfinance, no GPU --
# this is pure pandas and runs in seconds.
#
# Practical note: Days 1-6 each ran in their own Kaggle notebook, so
# their results/*.csv files are scattered across six separate outputs,
# not sitting in one working directory. Before running this: go to each
# prior notebook's Output panel, download its results/*.csv file(s),
# then either drop them in this notebook's working directory under
# results/, OR bundle them all into one Kaggle Dataset and add it as
# input -- the file search below checks both places automatically (it
# walks every /kaggle/input/*/ folder looking for the known filenames,
# so it doesn't matter what you named the dataset or how deep the CSVs
# ended up in it). Every check here is written to skip gracefully and
# say so when a file isn't found -- it won't crash on a partial set,
# it'll just tell you what's missing.
#
# Checklist below matches the plan's own Day 7 spec:
#   - all results internally consistent, no unexplained NaNs
#   - AIE confirmed on all 4 assets
#   - LIR > 1.0 for the augmentation models
#   - every statistical test has a p-value
#   - one master spreadsheet, everything in one place
#   - a flagged list of anything that looks like it needs a rerun


import os
import numpy as np
import pandas as pd

RESULTS_DIR = 'results'  # checked first; searched as a plain filename too

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 20)


def _search_roots():
    """Every place a result CSV could plausibly be sitting: the local
    results/ dir, the working directory itself, and every Kaggle input
    dataset (walked recursively, since an uploaded dataset's internal
    folder structure isn't something this script controls)."""
    roots = [RESULTS_DIR, '.']
    kaggle_input = '/kaggle/input'
    if os.path.isdir(kaggle_input):
        for dirpath, _dirnames, _filenames in os.walk(kaggle_input):
            roots.append(dirpath)
    return roots


def _path(*names):
    """Returns the first match for any of `names` across every search
    root, or None. Mirrors the flexible-filename pattern Days 2-4
    already use for Day 1's output, since different Day 1 script
    versions saved under different names."""
    for root in _search_roots():
        for name in names:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    return None


def load(label, *names):
    p = _path(*names)
    if p is None:
        print(f"  MISSING: {label} -- checked {list(names)}, none found")
        return None
    df = pd.read_csv(p)
    print(f"  loaded {label}: {p} ({len(df)} rows)")
    return df


print("="*70)
print("  DAY 7 — REVIEW + BUFFER")
print("="*70)
print("\nloading whatever result CSVs are present in "
      f"'{RESULTS_DIR}/'...\n")

day1 = load('Day 1 (baseline)', 'day1_baseline.csv', 'day1_cross_asset_results.csv')
day1_lir = load('Day 1 (LIR table)', 'day1_lir.csv', 'day1_lir_table.csv')
day2 = load('Day 2 (CycleGAN)', 'day2_cyclegan_results.csv')
day3 = load('Day 3 (WGAN-GP/SMOTE-TS)', 'day3_wgan_smote_results.csv')
day4_summary = load('Day 4 (validation summary)', 'day4_statistical_validation.csv')
day4_pvals = load('Day 4 (p-values)', 'day4_p_values_table.csv')
day5 = load('Day 5 (regime/RCAH)', 'day5_regime_analysis.csv')
day6 = load('Day 6 (QLSTM ablation)', 'day6_ablation_results.csv')

flags = []  # collects everything that looks off, printed as a summary at the end


# ── check 1: no unexplained NaNs ──────────────────────────────────────────────

print("\n" + "-"*70)
print("CHECK 1 — no unexplained NaN values")
print("-"*70)

for label, df, expected_nan_cols in [
    ('day1', day1, []),
    ('day2', day2, []),
    ('day3', day3, []),
    ('day4_summary', day4_summary, []),
    # McNemar_stat is only populated when Day 4's script uses the chi-square
    # approximation (>=25 discordant pairs) -- the exact binomial path
    # (the common case at these sample sizes) leaves it blank by design,
    # so it's not a real gap.
    ('day4_pvals', day4_pvals, ['McNemar_stat']),
    # day5's Real_DA/Hybrid_DA/AugBenefit are expected to be NaN for any
    # regime the skip-guard caught (COVID_Crash almost certainly) -- that's
    # documented behavior, not a bug, so it's excluded from the flag count.
    ('day5', day5, ['Real_DA', 'Hybrid_DA', 'AugBenefit']),
    ('day6', day6, []),
]:
    if df is None:
        continue
    check_cols = [c for c in df.columns if c not in expected_nan_cols]
    nan_counts = df[check_cols].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if len(bad) > 0:
        print(f"  {label}: unexpected NaNs in {dict(bad)}")
        flags.append(f"{label} has unexpected NaN values in: {list(bad.index)}")
    else:
        print(f"  {label}: no unexpected NaNs")
    if expected_nan_cols and df is not None:
        for col in expected_nan_cols:
            if col not in df.columns:
                continue
            n = df[col].isna().sum()
            if n:
                print(f"    ({n} expected NaN in '{col}' -- documented "
                      f"behavior for this column, not a bug)")


# ── check 2: AIE confirmed on all 4 assets ────────────────────────────────────

print("\n" + "-"*70)
print("CHECK 2 — AIE confirmed on all 4 assets (persistence R2 > 0.90)")
print("-"*70)

if day1 is not None and 'Persist_R2' in day1.columns:
    for _, row in day1.iterrows():
        r2 = row['Persist_R2']
        status = 'CONFIRMED' if r2 > 0.90 else 'NOT CONFIRMED'
        print(f"  {row['Asset']:8s}: persistence R2 = {r2:.4f}  -> {status}")
        if r2 <= 0.90:
            flags.append(f"AIE not confirmed for {row['Asset']} (persistence R2={r2:.4f})")
    if (day1['Persist_R2'] > 0.90).all():
        print("  all 4 assets confirmed.")
elif day1 is None:
    print("  SKIPPED: day1 results not loaded")
    flags.append("Could not check AIE -- day1 results missing")
else:
    print("  SKIPPED: day1 results loaded but no Persist_R2 column found")
    flags.append("day1 results present but missing Persist_R2 column")


# ── check 3: LIR > 1.0 for augmentation models ────────────────────────────────

print("\n" + "-"*70)
print("CHECK 3 — LIR > 1.0 for the augmentation models")
print("-"*70)

if day1_lir is not None:
    lir_col = next((c for c in day1_lir.columns if 'LIR' in c.upper() and 'DA' in c.upper()), None)
    if lir_col:
        for _, row in day1_lir.iterrows():
            model_col = day1_lir.columns[0]
            lir = row[lir_col]
            status = 'inflated (expected)' if lir > 1.0 else 'NOT inflated -- check this'
            print(f"  {row[model_col]:20s}: LIR = {lir:.2f}x  -> {status}")
            if lir <= 1.0:
                flags.append(f"LIR <= 1.0 for {row[model_col]} ({lir:.2f}x) -- unexpected, worth a second look")
    else:
        print("  SKIPPED: no LIR column found in the loaded file")
        flags.append("day1_lir file present but no recognizable LIR column")
else:
    print("  SKIPPED: day1 LIR table not loaded (checked day1_lir.csv, day1_lir_table.csv)")
    flags.append("Could not check LIR -- day1 LIR table missing")


# ── check 4: every statistical test has a p-value ────────────────────────────

print("\n" + "-"*70)
print("CHECK 4 — every Day 4 statistical test has a p-value")
print("-"*70)

if day4_pvals is not None:
    p_cols = [c for c in day4_pvals.columns if c.lower().endswith('_p') or c.lower() == 'p_value']
    missing_p = 0
    for _, row in day4_pvals.iterrows():
        for c in p_cols:
            if pd.isna(row[c]):
                missing_p += 1
                flags.append(f"{row.get('Asset', '?')}/{row.get('Method', '?')} missing {c}")
    if missing_p == 0:
        print(f"  all {len(day4_pvals)} rows have every p-value populated "
              f"(checked columns: {p_cols})")
    else:
        print(f"  {missing_p} missing p-value(s) found -- see flags below")
else:
    print("  SKIPPED: day4 p-values table not loaded")
    flags.append("Could not check p-values -- day4_p_values_table.csv missing")


# ── build the master spreadsheet ──────────────────────────────────────────────

print("\n" + "-"*70)
print("BUILDING MASTER RESULTS TABLE")
print("-"*70)

master_rows = []

if day1 is not None:
    da_col = 'DA_mean' if 'DA_mean' in day1.columns else None
    for _, row in day1.iterrows():
        master_rows.append({
            'Day': 1, 'Asset': row['Asset'], 'Method': 'baseline',
            'DA': row.get(da_col) if da_col else np.nan,
            'Delta_pp': 0.0, 'p_value': row.get('ttest_p', np.nan),
        })

if day2 is not None:
    for _, row in day2.iterrows():
        master_rows.append({
            'Day': 2, 'Asset': row['Asset'], 'Method': 'CycleGAN',
            'DA': row.get('CycleGAN_DA_mean', np.nan),
            'Delta_pp': row.get('Delta_pp', np.nan),
            'p_value': row.get('ttest_p', np.nan),
        })

if day3 is not None:
    for _, row in day3.iterrows():
        master_rows.append({
            'Day': 3, 'Asset': row['Asset'], 'Method': row.get('Method', '?'),
            'DA': row.get('DA_mean', np.nan),
            'Delta_pp': row.get('Delta_pp', np.nan),
            'p_value': row.get('ttest_p', np.nan),
        })

if day4_pvals is not None:
    for _, row in day4_pvals.iterrows():
        master_rows.append({
            'Day': 4, 'Asset': row.get('Asset', '?'),
            'Method': f"{row.get('Method', '?')} (10-trial)",
            'DA': row.get('Method_DA', np.nan),
            'Delta_pp': row.get('Delta_pp', np.nan),
            'p_value': row.get('TTest_p', np.nan),
        })

if day6 is not None:
    for _, row in day6.iterrows():
        master_rows.append({
            'Day': 6, 'Asset': row['Asset'], 'Method': 'QLSTM (hidden=32)',
            'DA': row.get('QLSTM_DA_mean', np.nan),
            'Delta_pp': row.get('DA_gap_pp', np.nan),
            'p_value': np.nan,
        })

master_df = pd.DataFrame(master_rows)
if not master_df.empty:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'day7_master_results.csv')
    master_df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path} ({len(master_df)} rows across "
          f"{master_df['Day'].nunique()} day(s) of results)")
    print("\n" + master_df.to_string(index=False))
else:
    print("\nNo result CSVs were found at all -- nothing to build. "
          "Upload the Day 1-6 result CSVs into this notebook first.")


# ── final summary ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  DAY 7 SUMMARY")
print("="*70)

loaded = sum(x is not None for x in [day1, day1_lir, day2, day3, day4_summary, day4_pvals, day5, day6])
print(f"\nfiles loaded: {loaded}/8 expected result files")

if flags:
    print(f"\n{len(flags)} item(s) flagged for review:")
    for f in flags:
        print(f"  - {f}")
    print("\nPhD-student gut check: do any of these look like a real problem,")
    print("or are they all explainable (e.g. day5's expected COVID_Crash NaNs)?")
    print("Anything you can't explain -> worth rerunning before Week 2.")
else:
    print("\nno flags. If everything here matches what you remember from each")
    print("day's run, Week 1 is done -- rest, then start Day 8 (paper structure).")

print("\nday 7 done. next: day 8 -- paper structure, abstract, intro")
