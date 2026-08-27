import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT CLEANING FUNCTIONS
# ============================================================

from src.cleaning import (
    identify_invalid_records,
    remove_invalid_records,
    remove_duplicates,
    impute_missing_values,
)


# ============================================================
# TEST INVALID RECORD DETECTION
# ============================================================

def test_identify_invalid_records():

    df = pd.DataFrame({
        "experience_years": [5, -3, 10],
        "age": [30, 45, 16],
        "projects_completed": [5, 10, 8],
        "training_hours": [20, 40, 60],
        "salary_lakh": [10, 15, -5]
    })

    invalid_mask = identify_invalid_records(df)

    assert invalid_mask.tolist() == [
        False,
        True,
        True
    ]


# ============================================================
# TEST INVALID RECORD REMOVAL
# ============================================================

def test_remove_invalid_records():

    df = pd.DataFrame({
        "experience_years": [5, -3, 10],
        "age": [30, 45, 16],
        "projects_completed": [5, 10, 8],
        "training_hours": [20, 40, 60],
        "salary_lakh": [10, 15, -5]
    })

    cleaned = remove_invalid_records(df)

    assert len(cleaned) == 1

    assert cleaned["experience_years"].iloc[0] == 5

    assert cleaned["salary_lakh"].iloc[0] == 10


# ============================================================
# TEST DUPLICATE REMOVAL
# ============================================================

def test_remove_duplicates():

    df = pd.DataFrame({
        "experience_years": [5, 5, 10],
        "age": [30, 30, 40],
        "projects_completed": [5, 5, 8],
        "training_hours": [20, 20, 60],
        "salary_lakh": [10, 10, 15]
    })

    cleaned = remove_duplicates(df)

    assert len(cleaned) == 2

    assert cleaned.duplicated().sum() == 0


# ============================================================
# TEST MISSING VALUE IMPUTATION
# ============================================================

def test_impute_missing_values():

    df = pd.DataFrame({
        "experience_years": [5, 10, None],
        "age": [30, None, 40],
        "projects_completed": [5, 8, None],
        "training_hours": [20, None, 60],
        "salary_lakh": [10, 15, 20]
    })

    cleaned = impute_missing_values(df)

    # No missing values should remain
    assert cleaned.isna().sum().sum() == 0

    # Median imputation checks
    assert cleaned["experience_years"].iloc[2] == 7.5

    assert cleaned["age"].iloc[1] == 35

    assert cleaned["projects_completed"].iloc[2] == 6.5

    assert cleaned["training_hours"].iloc[1] == 40


# ============================================================
# TEST COMPLETE CLEANING LOGIC
# ============================================================

def test_cleaning_produces_valid_data():

    df = pd.DataFrame({
        "experience_years": [5, -3, 10, 5],
        "age": [30, 45, 40, 30],
        "projects_completed": [5, 10, 8, 5],
        "training_hours": [20, 40, 60, 20],
        "salary_lakh": [10, 15, 15, 10]
    })

    # Remove invalid records
    df = remove_invalid_records(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Impute missing values
    df = impute_missing_values(df)

    # --------------------------------------------------------
    # Invalid values should be gone
    # --------------------------------------------------------

    assert (
        (df["experience_years"] < 0).sum()
        == 0
    )

    assert (
        (df["age"] < 18).sum()
        == 0
    )

    assert (
        (df["salary_lakh"] < 0).sum()
        == 0
    )

    # --------------------------------------------------------
    # Duplicates should be gone
    # --------------------------------------------------------

    assert df.duplicated().sum() == 0

    # --------------------------------------------------------
    # Missing values should be gone
    # --------------------------------------------------------

    assert df.isna().sum().sum() == 0