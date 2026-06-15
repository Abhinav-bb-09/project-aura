"""
src/aura/tools/profiler.py

Layer A of the Schema Discovery Agent: pure deterministic profiling.
No LLM calls here — pandas and numpy know the data better than any model.
This runs first, cheaply and instantly, before we spend a single API token.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def profile_dataset(source: str | Path) -> dict[str, Any]:
    """
    Load any CSV, Excel, or SQLite file and return a full schema profile.

    Parameters
    ----------
    source : str or Path
        Path to a .csv, .xlsx, .xls, or .sqlite / .db file.

    Returns
    -------
    dict with keys:
        source_path   : str
        file_type     : str
        tables        : dict[table_name -> TableProfile]
        relationships : list of candidate foreign-key pairs
        summary       : high-level numbers for display
    """
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(f"No file found at: {source}")

    ext = source.suffix.lower()

    if ext == ".csv":
        tables = _profile_csv(source)
        file_type = "csv"
    elif ext in (".xlsx", ".xls"):
        tables = _profile_excel(source)
        file_type = "excel"
    elif ext in (".sqlite", ".db", ".sqlite3"):
        tables = _profile_sqlite(source)
        file_type = "sqlite"
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use CSV, Excel, or SQLite.")

    relationships = _detect_relationships(tables)

    summary = {
        "total_tables": len(tables),
        "total_columns": sum(t["column_count"] for t in tables.values()),
        "total_rows": sum(t["row_count"] for t in tables.values()),
    }

    return {
        "source_path": str(source),
        "file_type": file_type,
        "tables": tables,
        "relationships": relationships,
        "summary": summary,
    }


# ─────────────────────────────────────────────
# FILE LOADERS
# ─────────────────────────────────────────────

def _profile_csv(path: Path) -> dict:
    """Load a single CSV and return it as a one-table profile dict."""
    # Try UTF-8 first; fall back to latin-1 for files with special characters.
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    table_name = path.stem  # filename without extension becomes the table name
    return {table_name: _profile_dataframe(df, table_name)}


def _profile_excel(path: Path) -> dict:
    """Load all sheets from an Excel file. Each sheet becomes one table."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    # sheet_name=None returns a dict of {sheet_name: DataFrame}
    return {name: _profile_dataframe(df, name) for name, df in sheets.items()}


def _profile_sqlite(path: Path) -> dict:
    """
    Load all user tables from a SQLite database.
    Reads each column individually to safely skip binary/blob columns,
    since SQLite's dynamic typing means declared type isn't reliable.
    """
    conn = sqlite3.connect(path)
    table_names = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()

    tables = {}
    for table_name in table_names:
        # Get all column names for this table
        pragma = pd.read_sql(
            f"PRAGMA table_info('{table_name}')", conn
        )
        all_cols = pragma["name"].tolist()

        # Probe each column individually — skip any that contain binary data
        safe_cols = []
        for col in all_cols:
            try:
                test = pd.read_sql(
                    f'SELECT "{col}" FROM "{table_name}" LIMIT 5', conn
                )
                # Try converting to string to catch hidden binary columns
                test[col].astype(str)
                safe_cols.append(col)
            except Exception:
                pass  # silently skip unreadable columns

        if not safe_cols:
            continue

        cols_sql = ", ".join(f'"{c}"' for c in safe_cols)
        df = pd.read_sql(f'SELECT {cols_sql} FROM "{table_name}"', conn)
        tables[table_name] = _profile_dataframe(df, table_name)

    conn.close()
    return tables


# ─────────────────────────────────────────────
# CORE PROFILER
# ─────────────────────────────────────────────

def _profile_dataframe(df: pd.DataFrame, table_name: str) -> dict:
    """
    Compute a full statistical profile for one DataFrame.
    This is the heart of Layer A.
    """
    columns = {}
    for col in df.columns:
        columns[col] = _profile_column(df[col])

    candidate_keys = _detect_candidate_keys(df)

    return {
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "candidate_keys": candidate_keys,
        "sample_rows": df.head(3).fillna("").to_dict(orient="records"),
    }


def _profile_column(series: pd.Series) -> dict:
    """
    Profile a single column. Returns different stats depending on dtype.
    We detect four kinds: numeric, datetime, boolean, and categorical/text.
    """
    total = len(series)
    null_count = series.isna().sum()
    null_pct = round(null_count / total * 100, 2) if total > 0 else 0
    distinct_count = series.nunique(dropna=True)
    distinct_pct = round(distinct_count / total * 100, 2) if total > 0 else 0

    # Base stats every column gets
    profile = {
        "dtype_raw": str(series.dtype),
        "semantic_type": None,       # filled in below
        "null_count": int(null_count),
        "null_pct": null_pct,
        "distinct_count": int(distinct_count),
        "distinct_pct": distinct_pct,
        "sample_values": _safe_sample_values(series),
    }

    # ── Numeric ──────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        profile["semantic_type"] = "numeric"
        clean = series.dropna()
        if len(clean) > 0:
            profile.update({
                "min": _safe_scalar(clean.min()),
                "max": _safe_scalar(clean.max()),
                "mean": round(float(clean.mean()), 4),
                "median": round(float(clean.median()), 4),
                "std": round(float(clean.std()), 4),
                # Skewness tells us if the distribution is lopsided
                # > 1 or < -1 usually means outliers worth flagging
                "skewness": round(float(clean.skew()), 4),
                "zeros_pct": round((clean == 0).sum() / len(clean) * 100, 2),
                "negative_pct": round((clean < 0).sum() / len(clean) * 100, 2),
            })

    # ── Datetime ─────────────────────────────
    elif pd.api.types.is_datetime64_any_dtype(series):
        profile["semantic_type"] = "datetime"
        clean = series.dropna()
        if len(clean) > 0:
            profile.update({
                "min": str(clean.min()),
                "max": str(clean.max()),
                "range_days": (clean.max() - clean.min()).days,
            })

    # ── Boolean ──────────────────────────────
    elif pd.api.types.is_bool_dtype(series):
        profile["semantic_type"] = "boolean"
        clean = series.dropna()
        if len(clean) > 0:
            profile["true_pct"] = round(clean.sum() / len(clean) * 100, 2)

    # ── Categorical / Text ───────────────────
    else:
        # Try to parse as datetime — pandas doesn't always auto-detect these
        profile["semantic_type"] = _infer_text_semantic(series)
        if profile["semantic_type"] == "categorical":
            # For low-cardinality text, value counts are very informative
            top = series.value_counts(dropna=True).head(5)
            profile["top_values"] = {
                str(k): int(v) for k, v in top.items()
            }

    return profile


def _infer_text_semantic(series: pd.Series) -> str:
    """
    Heuristics to label a text column's likely meaning.
    Order matters: check most specific patterns first.
    """
    try:
        sample = series.dropna().astype(str)
    except (UnicodeDecodeError, Exception):
        return "binary"

    if len(sample) == 0:
        return "text"

    # Try datetime parse on a sample — cheap way to detect date strings
    try:
        pd.to_datetime(sample.head(50), format="mixed")
        return "datetime_string"
    except Exception:
        pass

    # Low cardinality = likely categorical (e.g. status, region, gender)
    if series.nunique() <= 20 or series.nunique() / len(series) < 0.05:
        return "categorical"

    # High cardinality, long strings = likely free text
    avg_len = sample.str.len().mean()
    if avg_len > 50:
        return "text"

    # High cardinality, short strings = likely IDs or codes
    if series.nunique() / len(series) > 0.9:
        return "id_or_code"

    return "categorical"


# ─────────────────────────────────────────────
# KEY DETECTION
# ─────────────────────────────────────────────

def _detect_candidate_keys(df: pd.DataFrame) -> list[str]:
    """
    A column is a candidate primary key if it has:
    - Zero nulls
    - 100% distinct values (every row is unique)

    This is a pure statistical check — no guessing from column names.
    """
    candidates = []
    for col in df.columns:
        if df[col].isna().sum() == 0 and df[col].nunique() == len(df):
            candidates.append(col)
    return candidates


def _detect_relationships(tables: dict) -> list[dict]:
    """
    Detect candidate foreign key relationships across tables by checking
    whether the values in one column are a subset of another column's values.

    Only runs when there are 2+ tables (i.e. Excel multi-sheet or SQLite).
    Returns a list of relationship dicts for the Schema Agent to reason over.
    """
    if len(tables) < 2:
        return []

    relationships = []
    table_names = list(tables.keys())

    for i, t1 in enumerate(table_names):
        for t2 in table_names[i + 1:]:
            cols1 = tables[t1]["columns"]
            cols2 = tables[t2]["columns"]

            for c1 in cols1:
                for c2 in cols2:
                    # Only compare columns with compatible types
                    if cols1[c1]["semantic_type"] != cols2[c2]["semantic_type"]:
                        continue
                    # Skip columns with too many nulls to be meaningful
                    if cols1[c1]["null_pct"] > 20 or cols2[c2]["null_pct"] > 20:
                        continue
                    # Check value overlap using sample_values
                    # (full comparison would be done on actual data in production)
                    s1 = set(str(v) for v in cols1[c1]["sample_values"])
                    s2 = set(str(v) for v in cols2[c2]["sample_values"])
                    if len(s1) > 0 and len(s2) > 0:
                        overlap = len(s1 & s2) / min(len(s1), len(s2))
                        if overlap >= 0.5:
                            relationships.append({
                                "table_a": t1,
                                "column_a": c1,
                                "table_b": t2,
                                "column_b": c2,
                                "overlap_score": round(overlap, 2),
                            })

    return relationships


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _safe_sample_values(series: pd.Series, n: int = 5) -> list:
    """Return up to n non-null sample values, JSON-serializable."""
    try:
        samples = series.dropna().head(n).tolist()
        return [_safe_scalar(v) for v in samples]
    except Exception:
        return []


def _safe_scalar(val: Any) -> Any:
    """Convert numpy scalar types to native Python so JSON serialization works."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def print_profile(profile: dict) -> None:
    """Pretty-print a profile dict for notebook exploration."""
    print(f"\n{'='*60}")
    print(f"SOURCE : {profile['source_path']}")
    print(f"TYPE   : {profile['file_type'].upper()}")
    print(f"TABLES : {profile['summary']['total_tables']}")
    print(f"COLUMNS: {profile['summary']['total_columns']}")
    print(f"ROWS   : {profile['summary']['total_rows']:,}")
    print(f"{'='*60}")

    for table_name, table in profile["tables"].items():
        print(f"\n  TABLE: {table_name} ({table['row_count']:,} rows × {table['column_count']} cols)")
        if table["candidate_keys"]:
            print(f"  CANDIDATE KEYS: {table['candidate_keys']}")

        for col_name, col in table["columns"].items():
            null_flag = " ⚠ HIGH NULLS" if col["null_pct"] > 30 else ""
            print(
                f"    {col_name:<30} "
                f"{col['semantic_type']:<16} "
                f"distinct={col['distinct_count']:<6} "
                f"nulls={col['null_pct']}%"
                f"{null_flag}"
            )

    if profile["relationships"]:
        print(f"\n  DETECTED RELATIONSHIPS:")
        for r in profile["relationships"]:
            print(
                f"    {r['table_a']}.{r['column_a']} "
                f"↔ {r['table_b']}.{r['column_b']} "
                f"(overlap={r['overlap_score']})"
            )