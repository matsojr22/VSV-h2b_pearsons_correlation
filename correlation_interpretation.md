# Interpret Correlation Results Across Age Groups

## Analysis Summary

### Key Findings from Current Analysis (Intensity Sums)

#### Averaged Replicates Analysis (Most Reliable Method)

**No Normalization:**

- p3: r = 0.66, r² = 0.44, p = 0.075 (not significant)
- p12: r = 0.86, r² = 0.75, p = 0.006 (significant)
- p20: r = 0.94, r² = 0.88, p = 0.0006 (highly significant)
- adult: r = 0.66, r² = 0.44, p = 0.052 (not significant)

**Pattern:** Strong increase p3→p12→p20, then drop at adult

**Log Normalization:**

- p3: r = 0.63, p = 0.093 (ns)
- p12: r = 0.88, p = 0.004 (significant)
- p20: r = 0.77, p = 0.027 (significant)
- adult: r = 0.77, p = 0.016 (significant)

**Pattern:** Increase p3→p12, then plateaus p20→adult

### Comparison to Prior Analysis (Cell/Pixel Counts)

**Prior Results:**

- p3: r = 0.0655, p = 0.8775 (ns)
- p12: r = 0.3366, p = 0.415 (ns)
- p20: r = 0.6533, p = 0.0789 (ns)
- adult: r = 0.7363, p = 0.0373 (significant)

**Pattern:** Gradual increase across all ages, reaching significance at adult

## Key Observations

### 1. Overall Correlation Strength

- Current analysis shows **much stronger correlations** than prior analysis
- This suggests intensity sums may be more sensitive to the relationship, OR
- May indicate confounding by region size (larger regions = higher intensities)

### 2. Age-Dependent Trends

**Consistent Pattern (p3→p12→p20):**

- Both analyses show **increasing correlation** from p3 to p12 to p20
- Current analysis shows this trend more strongly and reaches significance earlier

**Divergence at Adult:**

- Prior analysis: Continued increase to r=0.74 (significant)
- Current analysis: Drop to r=0.66 (not significant)
- This discrepancy needs investigation

### 3. Statistical Significance

**Current Analysis:**

- Reaches significance at p12 (earlier than prior)
- Strongest at p20
- Loses significance at adult

**Prior Analysis:**

- Only significant at adult
- Gradual increase throughout

## Potential Explanations

### Why p60 Doesn’t Show as Most Correlated (Despite Prior Analysis on the Same Data)

The **prior analysis** (cell counts in H2B vs manually painted axon-positive pixels in VSV, area-normalized) and the **current intensity analysis** are both from the **exact same images and same dataset**. The prior gave strong adult correlation (r ≈ 0.74) and the effect is striking in the images. Reviewers asked for a version with more identical metrics, so intensity from those same image sets is used here.

The issue is **which data the pipeline is reading**. H2B intensity is loaded from summaries that have **p60** (identifiers M762, M763, M773, M776). The **VSV intensity summary file** currently used (`vsv_raw_summary.csv`) only contains **adult** rows with identifiers M608, M609, M610 — so there is no identifier overlap with the H2B p60 animals. The pipeline therefore computes “adult” correlation by comparing mean H2B (p60 animals) to mean VSV (M608/M609/M610), which are different identifiers. So either (1) the wrong VSV file is being used, or (2) VSV intensity for the p60 image set (same as H2B) lives in another file or was never added to the summary. **Fix:** point the pipeline at VSV intensity that comes from the same p60 image set (same identifiers as H2B adult/p60), or add that data into the VSV summary so “adult” uses the same animals for both modalities.

### Why Adult Shows Lower Correlation

1. **No Overlap Between H2B and VSV Adult Replicates (Primary Cause)**

- **H2B adult** (p60) uses identifiers **M762, M763, M773, M776** (4 replicates).
- **VSV adult** uses identifiers **M608, M609, M610** (3 replicates).
- There is **no shared identifier** between the two modalities for adult. The pipeline normalizes `p60` → `adult`, so both are treated as the same cohort, but:
  - **averaged_replicates** adult correlation is between mean H2B per region (over M762–M776) and mean VSV per region (over M608–M610)—i.e. **different animals** in each modality, so the correlation is not a within-animal relationship.
  - **complete_pairs** adult has **no pairs** (inner merge on identifier yields zero rows) and is therefore undefined for adult.
- Treat the low or odd adult r as **largely structural** (different cohorts), not solely due to outliers. See the adult raw-data diagnostic below for replicate-level outlier checks.

2. **Sample Size Differences**

- Adult has 9 regions vs 8 for others (more regions = more variance)
- Identifier composition differs by modality as above (p60 vs adult labels)

3. **Biological Maturity Effects**

- Adult brain may have different signal-to-noise characteristics
- Mature brain regions may have more variable expression patterns

4. **Measurement Artifacts**

- Adult samples may have different imaging conditions
- Background signal may vary more in adult samples

5. **Region Size Confounding**

- If adult regions vary more in size, raw intensity sums would be more confounded
- This would explain why cell/pixel counts (size-normalized) show better adult correlation

### Why Current Analysis Shows Stronger Correlations

1. **Region Size Confounding**

- Intensity sums scale with region size
- If H2B and VSV both scale with size, this creates artificial correlation
- Cell/pixel counts are inherently size-normalized

2. **Signal Amplification**

- Intensity sums may amplify true biological signal
- But also amplify measurement noise

3. **Different Biological Meaning**

- Intensity sums reflect total signal (size × density)
- Cell counts reflect cell number (more size-normalized)
- These measure different aspects of the biology

## Recommendations

### 1. Normalize by Region Size

- If region areas are available, normalize intensities by area
- This would make results more comparable to cell/pixel count analysis
- Would test if size confounding explains the adult drop

### 2. Focus on Averaged Replicates Analysis

- This method shows the clearest patterns
- Log normalization provides good balance (reduces heteroscedasticity)

### 3. Investigate Adult Drop

- **Run the adult raw-data diagnostic:** `python examine_adult_raw_data.py` (from repo root). This writes to `adult_raw_diagnostic/`:
  - `identifier_report.txt` — H2B vs VSV adult identifiers and overlap (none).
  - `adult_h2b_replicate_level.csv`, `adult_vsv_replicate_level.csv` — per-region, per-replicate intensity (area-normalized).
  - `adult_replicate_outliers.csv` — replicates flagged as outliers (z-score \|z\|>2 or IQR outside 1.5×IQR) within each region and modality.
  - Optional: `adult_boxplots_by_region.png`, `adult_region_means_scatter.png` (use default; omit `--no-plots` to generate).
- Compare identifier composition (p60 vs adult) — the diagnostic reports this explicitly.
- Examine if certain regions or replicates drive the adult correlation; use the replicate-level tables and outlier CSV.

### 4. Consider Biological Interpretation

- The p3→p12→p20 increase suggests developmental coupling
- Adult drop may indicate:
  - Maturation-related decoupling
  - Different regulatory mechanisms in mature brain
  - Measurement artifacts specific to adult samples

## Conclusion

The current analysis **confirms the increasing trend from p3→p12→p20** seen in prior work, but shows it more strongly. The **adult drop is unexpected** and may reflect:

- Measurement artifacts (size confounding, different sample characteristics)
- Biological reality (maturation-related changes)
- Sample composition differences

The intensity-based approach appears more sensitive but potentially more confounded by region size compared to cell/pixel counts.

## Adult raw data diagnostic

To examine adult data at replicate level and check for outliers in H2B or VSV:

```bash
python examine_adult_raw_data.py
```

Outputs (in `adult_raw_diagnostic/`):

- **identifier_report.txt** — H2B vs VSV adult identifiers and overlap (currently none).
- **adult_h2b_replicate_level.csv**, **adult_vsv_replicate_level.csv** — per-region, per-replicate area-normalized intensity.
- **adult_replicate_outliers.csv** — replicates flagged as outliers (z-score |z|>2 or IQR outside 1.5×IQR) within each region and modality.
- **adult_boxplots_by_region.png**, **adult_region_means_scatter.png** — boxplots per region and scatter of region means (use `--no-plots` to skip).

The script uses the same summary CSVs and region exclusion list as the correlation pipeline. Adult H2B and VSV have no shared identifiers; the averaged-replicate adult correlation is therefore between different animals in the two modalities.
