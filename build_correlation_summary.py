#!/usr/bin/env python3
"""
Build correlation_summary.html from result CSVs.
Shows prior reference, which dataset best matches it (by r_RMSE), and all datasets
ordered from closest to farthest with r_RMSE and per-age tables.
"""

import os
import math
from pathlib import Path

# Prior reference (cell/pixel count results)
PRIOR_AGES = ['p3', 'p12', 'p20', 'adult']
PRIOR_R = {'p3': 0.0655, 'p12': 0.3366, 'p20': 0.6533, 'adult': 0.7363}
PRIOR_P = {'p3': 0.8775, 'p12': 0.415, 'p20': 0.0789, 'adult': 0.0373}
PRIOR_SIGNIFICANCE = {'p3': 'ns', 'p12': 'ns', 'p20': 'ns', 'adult': 'significant'}


def load_dataset_results(results_dir: str):
    """Discover dataset names from correlation_analysis and load averaged_replicates summary.
    Includes any extra result dirs (e.g. human-verified) that have averaged_replicates/summary.csv."""
    try:
        from correlation_analysis import DATASET_RUNS
        names = list(dict.fromkeys([run[0] for run in DATASET_RUNS]))
    except Exception:
        names = []
    for name in os.listdir(results_dir):
        if name.startswith('.'):
            continue
        path = os.path.join(results_dir, name, 'averaged_replicates', 'summary.csv')
        if os.path.isfile(path) and name not in names:
            names.append(name)
    names.sort()

    datasets = {}
    for name in names:
        path = os.path.join(results_dir, name, 'averaged_replicates', 'summary.csv')
        if not os.path.isfile(path):
            continue
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if (r.get('normalization') or '').strip().lower() == 'none']
        by_age = {}
        for r in rows:
            age = (r.get('age') or '').strip()
            if not age:
                continue
            try:
                rr = float(r.get('r', float('nan')))
                p = float(r.get('p_value', float('nan')))
                r_sq = float(r.get('r_squared', float('nan')))
                n_regions = r.get('n_regions', '')
            except (TypeError, ValueError):
                continue
            by_age[age] = {'r': rr, 'p_value': p, 'r_squared': r_sq, 'n_regions': n_regions}
        if by_age:
            datasets[name] = by_age
    return datasets


def load_comparison_results(results_dir: str, dataset_name: str):
    """Load full comparison (all analysis_type × normalization) from normalized/comparison_summary.csv."""
    import csv
    path = os.path.join(results_dir, dataset_name, 'normalized', 'comparison_summary.csv')
    if not os.path.isfile(path):
        return []
    rows_by_method = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            at = (r.get('analysis_type') or '').strip()
            norm = (r.get('normalization') or '').strip()
            age = (r.get('age') or '').strip()
            if not at or not norm or not age:
                continue
            key = (at, norm)
            if key not in rows_by_method:
                rows_by_method[key] = {}
            try:
                rr = float(r.get('r', float('nan'))) if r.get('r', '').strip() else float('nan')
                p = float(r.get('p_value', float('nan'))) if r.get('p_value', '').strip() else float('nan')
            except (TypeError, ValueError):
                rr, p = float('nan'), float('nan')
            rows_by_method[key][age] = {'r': rr, 'p': p}
    # Build list of (analysis_type, normalization, per-age r/p)
    out = []
    for (at, norm), by_age in sorted(rows_by_method.items()):
        out.append({'analysis_type': at, 'normalization': norm, 'by_age': by_age})
    return out


def compute_r_rmse(by_age: dict) -> float:
    """RMSE of r vs prior over overlapping ages. by_age can be age -> {'r': x, ...} or age -> {'r': x, 'p_value': ...}."""
    diffs_sq = []
    for age in PRIOR_AGES:
        if age not in PRIOR_R or age not in by_age:
            continue
        r_val = by_age[age].get('r') if isinstance(by_age[age], dict) else by_age[age]
        if r_val != r_val:  # nan
            continue
        d = PRIOR_R[age] - r_val
        diffs_sq.append(d * d)
    if not diffs_sq:
        return float('nan')
    return math.sqrt(sum(diffs_sq) / len(diffs_sq))


def is_r_increasing(by_age: dict) -> bool:
    """True if all PRIOR_AGES present with valid r and r is non-decreasing over p3 -> p12 -> p20 -> adult."""
    r_vals = []
    for age in PRIOR_AGES:
        if age not in by_age:
            return False
        r = by_age[age].get('r') if isinstance(by_age[age], dict) else by_age[age]
        if r != r:  # nan
            return False
        r_vals.append(r)
    return r_vals == sorted(r_vals)


def fmt_num(x, digits=4):
    if x != x:  # nan
        return '—'
    return f'{x:.{digits}g}' if abs(x) < 1e-3 or abs(x) >= 1e4 else f'{x:.4f}'


def build_html(datasets_with_rmse: list, results_dir: str, full_comparisons: dict, best_positive_trend=None) -> str:
    """Build HTML body: prior table, closest match sentence, best with positive trend, then all datasets with full method list."""
    lines = []
    lines.append('<h1>Correlation Summary: Match to Prior Results</h1>')

    # Prior results (reference)
    lines.append('<h2>Prior results (reference)</h2>')
    lines.append('<p>Cell/pixel count correlations used as the reference to match.</p>')
    lines.append('<table>')
    lines.append('<thead><tr><th>Age</th><th>r</th><th>p</th><th>Significance</th></tr></thead><tbody>')
    for age in PRIOR_AGES:
        lines.append(f'<tr><td>{age}</td><td>{PRIOR_R[age]}</td><td>{PRIOR_P[age]}</td><td>{PRIOR_SIGNIFICANCE[age]}</td></tr>')
    lines.append('</tbody></table>')

    if not datasets_with_rmse:
        lines.append('<p>No dataset results found. Run <code>correlation_analysis.py</code> first.</p>')
        return '\n'.join(lines)

    best_name, best_rmse = datasets_with_rmse[0][0], datasets_with_rmse[0][1]
    lines.append('<h2>Closest match</h2>')
    lines.append(f'<p>The dataset that most closely matches the prior results is <strong>{best_name}</strong> (r_RMSE = {fmt_num(best_rmse)}).</p>')

    # Best match with positive trend (r increasing over p3 -> p12 -> p20 -> adult)
    lines.append('<h2>Best match with positive trend (r increasing)</h2>')
    if best_positive_trend:
        name, row, r_rmse = best_positive_trend
        at, norm = row['analysis_type'], row['normalization']
        lines.append(
            f'<p>The best match with r increasing over p3→p12→p20→adult is <strong>{name}</strong> — {at}, {norm} '
            f'(r_RMSE = {fmt_num(r_rmse)}).</p>'
        )
    else:
        lines.append('<p>No method has r increasing over all four ages; consider reviewing methods in the tables below.</p>')

    # All datasets (by match to prior) — closest to farthest
    lines.append('<h2>All datasets (by match to prior)</h2>')
    lines.append('<p>Datasets ordered from closest to farthest from the prior r values. Each shows r_RMSE and per-age r, r², p.</p>')

    for name, r_rmse, by_age in datasets_with_rmse:
        lines.append(f'<h3>{name}</h3>')
        lines.append(f'<p><strong>r_RMSE</strong> = {fmt_num(r_rmse)}</p>')
        lines.append('<table>')
        lines.append('<thead><tr><th>Age</th><th>r</th><th>r²</th><th>p_value</th><th>n_regions</th></tr></thead><tbody>')
        for age in PRIOR_AGES:
            if age in by_age:
                row = by_age[age]
                lines.append(
                    f'<tr><td>{age}</td><td>{fmt_num(row["r"])}</td><td>{fmt_num(row["r_squared"])}</td>'
                    f'<td>{fmt_num(row["p_value"])}</td><td>{row["n_regions"]}</td></tr>'
                )
            else:
                lines.append(f'<tr><td>{age}</td><td colspan="4">—</td></tr>')
        lines.append('</tbody></table>')

        # Full list: all computation methods (analysis_type × normalization) with r_RMSE and r increasing
        comparison_list = full_comparisons.get(name, [])
        if comparison_list:
            # Sort so r-increasing methods appear first, then by r_RMSE
            def method_sort_key(row):
                by_age = row['by_age']
                r_rmse = compute_r_rmse(by_age)
                inc = is_r_increasing(by_age)
                return (0 if inc else 1, float('inf') if r_rmse != r_rmse else r_rmse, row['analysis_type'], row['normalization'])
            comparison_list = sorted(comparison_list, key=method_sort_key)
            lines.append('<h4>All computation methods</h4>')
            lines.append('<table>')
            header = '<thead><tr><th>Analysis</th><th>Normalization</th><th>r_RMSE</th><th>r ↑</th>'
            for age in PRIOR_AGES:
                header += f'<th>{age} r</th><th>{age} p</th>'
            header += '</tr></thead><tbody>'
            lines.append(header)
            for row in comparison_list:
                at, norm = row['analysis_type'], row['normalization']
                by_age = row['by_age']
                r_rmse = compute_r_rmse(by_age)
                r_inc = 'Yes' if is_r_increasing(by_age) else 'No'
                line = f'<tr><td>{at}</td><td>{norm}</td><td>{fmt_num(r_rmse)}</td><td>{r_inc}</td>'
                for age in PRIOR_AGES:
                    if age in by_age:
                        line += f'<td>{fmt_num(by_age[age]["r"])}</td><td>{fmt_num(by_age[age]["p"])}</td>'
                    else:
                        line += '<td>—</td><td>—</td>'
                line += '</tr>'
                lines.append(line)
            lines.append('</tbody></table>')

    return '\n'.join(lines)


def main(results_dir: str = 'results', output_path: str = 'correlation_summary.html'):
    base = Path(__file__).resolve().parent
    results_dir = base / results_dir
    output_path = base / output_path

    datasets = load_dataset_results(str(results_dir))
    if not datasets:
        print('No dataset summary CSVs found under', results_dir)
        return

    # Compute r_RMSE and sort closest to farthest
    with_rmse = []
    for name, by_age in datasets.items():
        r_rmse = compute_r_rmse(by_age)
        with_rmse.append((name, r_rmse, by_age))
    with_rmse.sort(key=lambda x: (float('inf') if x[1] != x[1] else x[1], x[0]))

    full_comparisons = {}
    for name in datasets.keys():
        full_comparisons[name] = load_comparison_results(str(results_dir), name)

    # Best match with positive trend: lowest r_RMSE among methods with r non-decreasing over all four ages
    positive_trend_candidates = []
    for name, comparison_list in full_comparisons.items():
        for row in comparison_list:
            by_age = row['by_age']
            r_rmse = compute_r_rmse(by_age)
            if is_r_increasing(by_age):
                positive_trend_candidates.append((name, row, r_rmse))
    positive_trend_candidates.sort(key=lambda x: (float('inf') if x[2] != x[2] else x[2], x[0], x[1]['analysis_type'], x[1]['normalization']))
    best_positive_trend = positive_trend_candidates[0] if positive_trend_candidates else None

    body = build_html(with_rmse, str(results_dir), full_comparisons, best_positive_trend=best_positive_trend)
    from html_template import generate_html
    html = generate_html(body, title='Correlation Summary')

    output_path.write_text(html, encoding='utf-8')
    print('Wrote', output_path)
    print('Closest match:', with_rmse[0][0], 'r_RMSE =', fmt_num(with_rmse[0][1]))
    if best_positive_trend:
        name, row, r_rmse = best_positive_trend
        print('Best with r increasing:', name, f"{row['analysis_type']}, {row['normalization']}", 'r_RMSE =', fmt_num(r_rmse))
    else:
        print('Best with r increasing: none (no method has r increasing over all four ages)')


if __name__ == '__main__':
    main()
