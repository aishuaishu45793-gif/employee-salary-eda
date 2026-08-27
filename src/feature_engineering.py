from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "employee_salary_cleaned.csv"
)

OUTPUTS_DIR = (
    PROJECT_ROOT
    / "outputs"
)

MODEL_READY_PATH = (
    OUTPUTS_DIR
    / "employee_salary_model_ready.csv"
)

X_PATH = (
    OUTPUTS_DIR
    / "X.csv"
)

Y_PATH = (
    OUTPUTS_DIR
    / "y.csv"
)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

def load_cleaned_data():
    """
    Load the cleaned employee salary dataset.
    """

    return pd.read_csv(
        CLEANED_DATA_PATH
    )


# ============================================================
# CREATE EXPERIENCE SCORE
# ============================================================

def create_experience_score(df):
    """
    Create normalized experience score.

    Formula:
        experience_score = experience_years / 20

    Expected range:
        0 to 1
    """

    df = df.copy()

    df["experience_score"] = (
        df["experience_years"] / 20
    )

    return df


# ============================================================
# CREATE PROJECT EXPERIENCE RATIO
# ============================================================

def create_project_experience_ratio(df):
    """
    Measure project productivity relative to experience.

    Formula:
        projects_completed /
        (experience_years + 1)

    +1 prevents division by zero.
    """

    df = df.copy()

    df["project_experience_ratio"] = (
        df["projects_completed"]
        / (df["experience_years"] + 1)
    )

    return df


# ============================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================

def create_features(df):
    """
    Create all engineered features.
    """

    df = create_experience_score(df)

    df = create_project_experience_ratio(df)

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(df):
    """
    Validate engineered features.
    """

    print("\n" + "=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)

    required_features = [
        "experience_score",
        "project_experience_ratio"
    ]

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    print("\nRequired features:")

    for feature in required_features:

        if feature in df.columns:
            print(f"{feature:30} : PASS")
        else:
            print(f"{feature:30} : FAIL")

    # --------------------------------------------------------
    # Experience score validation
    # --------------------------------------------------------

    experience_score_valid = (
        df["experience_score"].between(
            0,
            1
        ).all()
    )

    print(
        "\nExperience score range       : "
        f"{'PASS' if experience_score_valid else 'FAIL'}"
    )

    print(
        f"Minimum experience_score     : "
        f"{df['experience_score'].min():.4f}"
    )

    print(
        f"Maximum experience_score     : "
        f"{df['experience_score'].max():.4f}"
    )

    # --------------------------------------------------------
    # Ratio validation
    # --------------------------------------------------------

    ratio_is_finite = np.isfinite(
        df["project_experience_ratio"]
    ).all()

    ratio_is_non_negative = (
        df["project_experience_ratio"] >= 0
    ).all()

    print(
        "\nProject/experience ratio finite: "
        f"{'PASS' if ratio_is_finite else 'FAIL'}"
    )

    print(
        "Project/experience ratio >= 0  : "
        f"{'PASS' if ratio_is_non_negative else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    feature_missing = (
        df[required_features]
        .isna()
        .sum()
        .sum()
    )

    print(
        "\nMissing engineered values     : "
        f"{feature_missing}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validation_passed = (
        all(
            feature in df.columns
            for feature in required_features
        )
        and experience_score_valid
        and ratio_is_finite
        and ratio_is_non_negative
        and feature_missing == 0
    )

    print("\n" + "-" * 70)

    if validation_passed:
        print("FEATURE VALIDATION STATUS: PASS")
    else:
        print("FEATURE VALIDATION STATUS: FAIL")

    return validation_passed


# ============================================================
# CREATE MODEL-READY DATA
# ============================================================

def create_model_ready_data(df):
    """
    Separate predictors X and target y.

    Target:
        salary_lakh
    """

    feature_columns = [
        "experience_years",
        "age",
        "projects_completed",
        "training_hours",
        "experience_score",
        "project_experience_ratio"
    ]

    target_column = "salary_lakh"

    X = df[feature_columns].copy()

    y = df[[target_column]].copy()

    return X, y


# ============================================================
# VALIDATE X AND Y
# ============================================================

def validate_model_ready_data(X, y):
    """
    Validate X/y separation and prevent target leakage.
    """

    print("\n" + "=" * 70)
    print("MODEL-READY DATA VALIDATION")
    print("=" * 70)

    print(f"\nX rows     : {len(X)}")
    print(f"X columns  : {len(X.columns)}")
    print(f"y rows     : {len(y)}")
    print(f"y columns  : {len(y.columns)}")

    print("\nX features:")
    for column in X.columns:
        print(f"  - {column}")

    print("\nTarget:")
    print(f"  - {y.columns[0]}")

    # --------------------------------------------------------
    # Check row alignment
    # --------------------------------------------------------

    row_alignment = (
        len(X) == len(y)
    )

    print(
        "\nX/y row alignment : "
        f"{'PASS' if row_alignment else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Target leakage check
    # --------------------------------------------------------

    target_leakage = (
        "salary_lakh" in X.columns
    )

    print(
        "Target leakage    : "
        f"{'FAIL' if target_leakage else 'PASS'}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_X = X.isna().sum().sum()
    missing_y = y.isna().sum().sum()

    print(
        f"Missing values X  : {missing_X}"
    )

    print(
        f"Missing values y  : {missing_y}"
    )

    validation_passed = (
        row_alignment
        and not target_leakage
        and missing_X == 0
        and missing_y == 0
    )

    print("\n" + "-" * 70)

    if validation_passed:
        print("MODEL-READY VALIDATION STATUS: PASS")
    else:
        print("MODEL-READY VALIDATION STATUS: FAIL")

    return validation_passed


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(df, X, y):
    """
    Save model-ready dataset, X and y.
    """

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        MODEL_READY_PATH,
        index=False
    )

    X.to_csv(
        X_PATH,
        index=False
    )

    y.to_csv(
        Y_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"\nModel-ready data:\n{MODEL_READY_PATH}"
    )

    print(
        f"\nX data:\n{X_PATH}"
    )

    print(
        f"\ny data:\n{Y_PATH}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("EMPLOYEE SALARY — FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load cleaned data
    # --------------------------------------------------------

    df = load_cleaned_data()

    print(
        f"\nCleaned dataset rows    : {len(df)}"
    )

    print(
        f"Cleaned dataset columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # 2. Create engineered features
    # --------------------------------------------------------

    df = create_features(df)

    print("\nEngineered features created:")

    print(
        "  - experience_score"
    )

    print(
        "  - project_experience_ratio"
    )

    # --------------------------------------------------------
    # 3. Validate features
    # --------------------------------------------------------

    features_valid = validate_features(df)

    if not features_valid:

        print(
            "\nFeature engineering stopped "
            "because validation failed."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # 4. Create X and y
    # --------------------------------------------------------

    X, y = create_model_ready_data(df)

    # --------------------------------------------------------
    # 5. Validate X and y
    # --------------------------------------------------------

    model_data_valid = validate_model_ready_data(
        X,
        y
    )

    if not model_data_valid:

        print(
            "\nModel-ready data creation stopped "
            "because validation failed."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # 6. Save outputs
    # --------------------------------------------------------

    save_outputs(
        df,
        X,
        y
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)