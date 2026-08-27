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
# IMPORT FEATURE FUNCTIONS
# ============================================================

from src.feature_engineering import (
    create_experience_score,
    create_project_experience_ratio,
    create_features,
    create_model_ready_data,
)


# ============================================================
# TEST EXPERIENCE SCORE
# ============================================================

def test_experience_score():

    df = pd.DataFrame({
        "experience_years": [0, 5, 10, 20]
    })

    result = create_experience_score(df)

    assert "experience_score" in result.columns

    assert result["experience_score"].tolist() == [
        0.0,
        0.25,
        0.5,
        1.0
    ]


# ============================================================
# TEST PROJECT EXPERIENCE RATIO
# ============================================================

def test_project_experience_ratio():

    df = pd.DataFrame({
        "experience_years": [0, 4, 9],
        "projects_completed": [5, 10, 20]
    })

    result = create_project_experience_ratio(df)

    assert "project_experience_ratio" in result.columns

    assert result["project_experience_ratio"].iloc[0] == 5.0

    assert result["project_experience_ratio"].iloc[1] == 2.0

    assert result["project_experience_ratio"].iloc[2] == 2.0


# ============================================================
# TEST COMPLETE FEATURE CREATION
# ============================================================

def test_create_features():

    df = pd.DataFrame({
        "experience_years": [2, 5, 10],
        "age": [25, 30, 40],
        "projects_completed": [4, 10, 20],
        "training_hours": [20, 40, 60],
        "salary_lakh": [6, 10, 18]
    })

    result = create_features(df)

    assert "experience_score" in result.columns

    assert "project_experience_ratio" in result.columns

    assert result.isna().sum().sum() == 0


# ============================================================
# TEST MODEL-READY X AND Y
# ============================================================

def test_model_ready_data():

    df = pd.DataFrame({
        "experience_years": [2, 5, 10],
        "age": [25, 30, 40],
        "projects_completed": [4, 10, 20],
        "training_hours": [20, 40, 60],
        "salary_lakh": [6, 10, 18]
    })

    df = create_features(df)

    X, y = create_model_ready_data(df)

    # X should contain 6 predictor columns
    assert X.shape == (3, 6)

    # y should contain one target column
    assert y.shape == (3, 1)

    # Target should be salary_lakh
    assert list(y.columns) == [
        "salary_lakh"
    ]

    # salary_lakh must NOT be inside X
    assert "salary_lakh" not in X.columns

    # Required engineered features
    assert "experience_score" in X.columns

    assert "project_experience_ratio" in X.columns

    # X and y must have the same number of rows
    assert len(X) == len(y)


# ============================================================
# TEST TARGET LEAKAGE
# ============================================================

def test_no_target_leakage():

    df = pd.DataFrame({
        "experience_years": [2, 5],
        "age": [25, 30],
        "projects_completed": [4, 10],
        "training_hours": [20, 40],
        "salary_lakh": [6, 10]
    })

    df = create_features(df)

    X, y = create_model_ready_data(df)

    assert "salary_lakh" not in X.columns


# ============================================================
# TEST EXPERIENCE SCORE RANGE
# ============================================================

def test_experience_score_range():

    df = pd.DataFrame({
        "experience_years": [0, 5, 10, 15, 20]
    })

    result = create_experience_score(df)

    assert (
        result["experience_score"]
        .between(0, 1)
        .all()
    )