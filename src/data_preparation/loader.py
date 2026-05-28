"""
src/data_preparation/loader.py
-------------------------------
Handles loading raw loan data. Supports:
  - Single CSV / Excel file
  - Merging multiple Excel sheets (mirrors real-world loan data delivery)
"""

import pandas as pd
from pathlib import Path
from typing import Union, List, Optional


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a single CSV file and return a DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run  python data/generate_sample_data.py  first."
        )
    df = pd.read_csv(path)
    print(f"[Loader] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols from {path.name}")
    return df


def load_excel_sheets(
    filepath: Union[str, Path],
    sheets: Optional[List[str]] = None,
    merge_on: str = "user_id"
) -> pd.DataFrame:
    """
    Load one or more sheets from an Excel workbook and merge them on a
    common key.  Useful when loan applicant data is split across tabs
    (e.g. 'Demographics', 'LoanDetails', 'CreditHistory').

    Parameters
    ----------
    filepath  : Path to the .xlsx file
    sheets    : List of sheet names to load. None = load all sheets.
    merge_on  : Column used to join sheets together.
    """
    path = Path(filepath)
    xl = pd.ExcelFile(path)
    sheet_names = sheets or xl.sheet_names

    frames = {}
    for name in sheet_names:
        frames[name] = xl.parse(name)
        print(f"[Loader] Sheet '{name}' → {frames[name].shape}")

    # Iteratively merge all sheets on the common key
    merged = list(frames.values())[0]
    for sheet_name, frame in list(frames.items())[1:]:
        merged = merged.merge(frame, on=merge_on, how="outer", suffixes=("", f"_{sheet_name}"))

    print(f"[Loader] Merged shape: {merged.shape}")
    return merged


def quick_eda(df: pd.DataFrame) -> None:
    """Print a concise summary of the loaded dataset."""
    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows       : {df.shape[0]:,}")
    print(f"  Columns    : {df.shape[1]}")
    print(f"  Duplicates : {df.duplicated().sum()}")
    print(f"  Memory     : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    print("\n  Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<30} {dtype}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\n  Missing Values:")
        for col, cnt in missing.items():
            print(f"    {col:<30} {cnt:>5} ({cnt / len(df):.1%})")
    else:
        print("\n  Missing Values: None")

    if "Defaulter" in df.columns:
        counts = df["Defaulter"].value_counts()
        print(f"\n  Target Distribution:")
        print(f"    No Default (0) : {counts.get(0, 0):,}  ({counts.get(0, 0)/len(df):.1%})")
        print(f"    Default    (1) : {counts.get(1, 0):,}  ({counts.get(1, 0)/len(df):.1%})")

    print("=" * 60 + "\n")
