"""
src/aura/tools/stats.py

Deterministic statistical analysis layer. No LLM calls.
Takes a DataFrame (from the Engineer Agent) and returns
structured statistical findings with real p-values.

This is what separates Aura from systems that just describe
data with words — we run actual tests and report actual numbers.
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def analyze_dataframe(df: pd.DataFrame, question: str = "") -> dict[str, Any]:
    """
    Run a full statistical analysis on a DataFrame.

    Parameters
    ----------
    df       : the query result from the Engineer Agent
    question : the original user question (used to pick relevant tests)

    Returns
    -------
    dict with keys:
        shape           : (rows, cols)
        numeric_stats   : descriptive stats per numeric column
        distributions   : normality test results
        correlations    : Pearson correlation matrix for numeric cols
        outliers        : outlier flags per numeric column
        categorical_stats: value counts for categorical columns
        confidence_score: 0-100 score based on data quality
        confidence_factors: explanation of what drove the score
        findings        : list of auto-detected interesting patterns
    """
    if df.empty:
        return {"error": "Empty DataFrame — no data to analyze"}

    result = {
        "shape": df.shape,
        "question": question,
        "numeric_stats": {},
        "distributions": {},
        "correlations": {},
        "outliers": {},
        "categorical_stats": {},
        "confidence_score": 0,
        "confidence_factors": [],
        "findings": [],
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── Numeric analysis ──────────────────────────────────────
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        result["numeric_stats"][col] = _describe_numeric(series)
        result["distributions"][col] = _test_normality(series)
        result["outliers"][col]      = _detect_outliers(series)

    # ── Correlation matrix ────────────────────────────────────
    if len(numeric_cols) >= 2:
        result["correlations"] = _compute_correlations(df[numeric_cols])

    # ── Categorical analysis ──────────────────────────────────
    for col in categorical_cols:
        result["categorical_stats"][col] = _describe_categorical(df[col])

    # ── Auto-detect interesting findings ─────────────────────
    result["findings"] = _detect_findings(df, result, numeric_cols, categorical_cols)

    # ── Confidence score ──────────────────────────────────────
    score, factors = _compute_confidence(df, result, numeric_cols)
    result["confidence_score"] = score
    result["confidence_factors"] = factors

    return result


# ─────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────

def _describe_numeric(series: pd.Series) -> dict:
    """Full descriptive statistics for a numeric column."""
    return {
        "count"   : int(len(series)),
        "mean"    : round(float(series.mean()), 4),
        "median"  : round(float(series.median()), 4),
        "std"     : round(float(series.std()), 4),
        "min"     : round(float(series.min()), 4),
        "max"     : round(float(series.max()), 4),
        "q25"     : round(float(series.quantile(0.25)), 4),
        "q75"     : round(float(series.quantile(0.75)), 4),
        "skewness": round(float(series.skew()), 4),
        "kurtosis": round(float(series.kurtosis()), 4),
        "cv"      : round(float(series.std() / series.mean()), 4) if series.mean() != 0 else None,
    }


def _test_normality(series: pd.Series) -> dict:
    """
    Shapiro-Wilk test for normality (best for n < 5000).
    p > 0.05 means we cannot reject normality.

    Why this matters: many downstream statistical tests assume
    normally distributed data. Flagging non-normality early
    prevents the Scientist from drawing invalid conclusions.
    """
    if len(series) < 3:
        return {"test": "insufficient_data", "is_normal": None}

    # Shapiro-Wilk is most powerful for small-medium samples
    # For large samples, use D'Agostino-Pearson
    if len(series) <= 5000:
        stat, p_value = scipy_stats.shapiro(series)
        test_name = "shapiro-wilk"
    else:
        stat, p_value = scipy_stats.normaltest(series)
        test_name = "dagostino-pearson"

    return {
        "test"     : test_name,
        "statistic": round(float(stat), 4),
        "p_value"  : round(float(p_value), 4),
        "is_normal": bool(p_value > 0.05),
    }


def _detect_outliers(series: pd.Series) -> dict:
    """
    IQR-based outlier detection.
    Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are outliers.
    This is the standard Tukey fence method — robust to non-normality.
    """
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25

    lower_fence = q25 - 1.5 * iqr
    upper_fence = q75 + 1.5 * iqr

    outliers = series[(series < lower_fence) | (series > upper_fence)]

    return {
        "count"      : int(len(outliers)),
        "pct"        : round(len(outliers) / len(series) * 100, 2),
        "lower_fence": round(float(lower_fence), 4),
        "upper_fence": round(float(upper_fence), 4),
        "has_outliers": len(outliers) > 0,
    }


def _compute_correlations(numeric_df: pd.DataFrame) -> dict:
    """
    Pearson correlation matrix. Only includes pairs with
    abs(r) > 0.3 to avoid noise — weak correlations aren't
    actionable and just clutter the output.
    """
    if len(numeric_df) < 3:
        return {}

    corr_matrix = numeric_df.corr(method="pearson")
    strong_pairs = []

    cols = corr_matrix.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i+1:]:
            r = corr_matrix.loc[col_a, col_b]
            if abs(r) > 0.3 and not np.isnan(r):
                strong_pairs.append({
                    "col_a"    : col_a,
                    "col_b"    : col_b,
                    "r"        : round(float(r), 4),
                    "strength" : _correlation_label(r),
                })

    return {
        "matrix": corr_matrix.round(3).to_dict(),
        "strong_pairs": sorted(strong_pairs, key=lambda x: abs(x["r"]), reverse=True),
    }


def _describe_categorical(series: pd.Series) -> dict:
    """Value counts and concentration metrics for a categorical column."""
    counts = series.value_counts(dropna=True)
    total  = len(series.dropna())

    return {
        "unique_count": int(series.nunique()),
        "top_values"  : {str(k): int(v) for k, v in counts.head(10).items()},
        "top_pct"     : round(counts.iloc[0] / total * 100, 2) if len(counts) > 0 else 0,
        # High concentration = one value dominates = potentially low signal
        "concentrated": bool(counts.iloc[0] / total > 0.8) if len(counts) > 0 else False,
    }


# ─────────────────────────────────────────────
# AUTO-FINDING DETECTION
# ─────────────────────────────────────────────

def _detect_findings(
    df: pd.DataFrame,
    result: dict,
    numeric_cols: list,
    categorical_cols: list,
) -> list[dict]:
    """
    Automatically surface interesting patterns.
    Each finding has a type, description, and severity.
    These become the seeds for the Scientist Agent's LLM interpretation.
    """
    findings = []

    # ── High outlier rate ─────────────────────────────────────
    for col, outlier_info in result["outliers"].items():
        if outlier_info["pct"] > 5:
            findings.append({
                "type"       : "outliers",
                "column"     : col,
                "severity"   : "high" if outlier_info["pct"] > 15 else "medium",
                "description": f"{col} has {outlier_info['pct']}% outliers "
                               f"(outside [{outlier_info['lower_fence']}, {outlier_info['upper_fence']}])",
            })

    # ── Strong correlations ───────────────────────────────────
    strong_pairs = result["correlations"].get("strong_pairs", [])
    for pair in strong_pairs[:3]:  # top 3 only
        findings.append({
            "type"       : "correlation",
            "severity"   : "high" if abs(pair["r"]) > 0.7 else "medium",
            "description": f"{pair['col_a']} and {pair['col_b']} are "
                           f"{pair['strength']} correlated (r={pair['r']})",
        })

    # ── High skewness ─────────────────────────────────────────
    for col, stats in result["numeric_stats"].items():
        if abs(stats["skewness"]) > 2:
            findings.append({
                "type"       : "skewness",
                "column"     : col,
                "severity"   : "medium",
                "description": f"{col} is heavily skewed ({stats['skewness']:.2f}), "
                               f"suggesting a long tail in the distribution",
            })

    # ── Top value dominance ───────────────────────────────────
    for col, cat_stats in result["categorical_stats"].items():
        if cat_stats["concentrated"]:
            top_val = list(cat_stats["top_values"].keys())[0]
            findings.append({
                "type"       : "concentration",
                "column"     : col,
                "severity"   : "low",
                "description": f"{col} is dominated by '{top_val}' "
                               f"({cat_stats['top_pct']}% of values)",
            })

    # ── Small sample warning ──────────────────────────────────
    if len(df) < 30:
        findings.append({
            "type"       : "sample_size",
            "severity"   : "high",
            "description": f"Only {len(df)} rows — statistical conclusions may be unreliable",
        })

    return findings


# ─────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────

def _compute_confidence(
    df: pd.DataFrame,
    result: dict,
    numeric_cols: list,
) -> tuple[int, list[str]]:
    """
    Compute a 0-100 confidence score for this analysis.

    Factors (each can add or subtract points):
    - Sample size         : more rows = higher confidence
    - Outlier rate        : many outliers reduce confidence
    - Data completeness   : nulls reduce confidence
    - Distribution normality: normal distributions = more reliable stats
    """
    score   = 50  # start at 50 (neutral)
    factors = []

    # ── Sample size ───────────────────────────────────────────
    n = len(df)
    if n >= 1000:
        score += 25
        factors.append(f"+25: large sample (n={n:,})")
    elif n >= 100:
        score += 15
        factors.append(f"+15: adequate sample (n={n:,})")
    elif n >= 30:
        score += 5
        factors.append(f"+5: small but usable sample (n={n:,})")
    else:
        score -= 20
        factors.append(f"-20: very small sample (n={n:,}), low reliability")

    # ── Outlier penalty ───────────────────────────────────────
    if result["outliers"]:
        avg_outlier_pct = np.mean([v["pct"] for v in result["outliers"].values()])
        if avg_outlier_pct > 15:
            score -= 15
            factors.append(f"-15: high average outlier rate ({avg_outlier_pct:.1f}%)")
        elif avg_outlier_pct > 5:
            score -= 5
            factors.append(f"-5: moderate outlier rate ({avg_outlier_pct:.1f}%)")

    # ── Null penalty ──────────────────────────────────────────
    null_pct = df.isnull().mean().mean() * 100
    if null_pct > 20:
        score -= 15
        factors.append(f"-15: high null rate ({null_pct:.1f}%)")
    elif null_pct > 5:
        score -= 5
        factors.append(f"-5: moderate null rate ({null_pct:.1f}%)")
    else:
        score += 5
        factors.append(f"+5: low null rate ({null_pct:.1f}%)")

    # ── Normality bonus ───────────────────────────────────────
    if result["distributions"]:
        normal_pct = np.mean([
            v["is_normal"] for v in result["distributions"].values()
            if v.get("is_normal") is not None
        ]) * 100
        if normal_pct > 50:
            score += 10
            factors.append(f"+10: majority of numeric columns are normally distributed")

    # Clamp to [0, 100]
    score = max(0, min(100, score))
    return int(score), factors


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _correlation_label(r: float) -> str:
    """Convert Pearson r to a human-readable strength label."""
    abs_r = abs(r)
    direction = "positively" if r > 0 else "negatively"
    if abs_r >= 0.9:
        return f"very strongly {direction}"
    elif abs_r >= 0.7:
        return f"strongly {direction}"
    elif abs_r >= 0.5:
        return f"moderately {direction}"
    else:
        return f"weakly {direction}"