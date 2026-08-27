from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "03_employee_salary.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CLEANED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "employee_salary_cleaned.csv"
)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():
    """
    Load the original employee salary dataset.

    The raw CSV is never modified.
    """
    return pd.read_csv(RAW_DATA_PATH)


# ============================================================
# IDENTIFY INVALID RECORDS
# ============================================================

def identify_invalid_records(df):
    """
    Identify records violating the project domain rules.

    Rules:
    - age must be at least 18
    - experience_years cannot be negative
    - salary_lakh cannot be negative
    """

    invalid_age = df["age"] < 18

    invalid_experience = (
        df["experience_years"] < 0
    )

    invalid_salary = (
        df["salary_lakh"] < 0
    )

    invalid_mask = (
        invalid_age
        | invalid_experience
        | invalid_salary
    )

    return invalid_mask


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

def remove_invalid_records(df):
    """
    Remove records containing invalid domain values.
    """

    invalid_mask = identify_invalid_records(df)

    cleaned_df = df.loc[
        ~invalid_mask
    ].copy()

    return cleaned_df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """
    Remove exact duplicate rows.
    """

    cleaned_df = df.drop_duplicates(
        keep="first"
    ).copy()

    return cleaned_df


# ============================================================
# IMPUTE MISSING VALUES
# ============================================================

def impute_missing_values(df):
    """
    Impute missing numerical values using column medians.

    Median imputation is used because it is less sensitive
    to extreme values than mean imputation.
    """

    cleaned_df = df.copy()

    numeric_columns = [
        "experience_years",
        "age",
        "projects_completed",
        "training_hours",
        "salary_lakh"
    ]

    for column in numeric_columns:

        if cleaned_df[column].isna().any():

            median_value = cleaned_df[column].median()

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna(median_value)
            )

    return cleaned_df


# ============================================================
# VALIDATE CLEANED DATA
# ============================================================

def validate_cleaned_data(df):
    """
    Validate that the cleaned dataset satisfies
    the project data-quality requirements.
    """

    print("\n" + "=" * 70)
    print("POST-CLEANING VALIDATION")
    print("=" * 70)

    missing_cells = df.isna().sum().sum()

    duplicate_rows = df.duplicated().sum()

    invalid_age = (
        df["age"] < 18
    ).sum()

    negative_experience = (
        df["experience_years"] < 0
    ).sum()

    negative_salary = (
        df["salary_lakh"] < 0
    ).sum()

    print(f"Rows                  : {len(df)}")
    print(f"Columns               : {len(df.columns)}")
    print(f"Missing cells         : {missing_cells}")
    print(f"Duplicate rows        : {duplicate_rows}")
    print(f"Invalid age           : {invalid_age}")
    print(f"Negative experience   : {negative_experience}")
    print(f"Negative salary       : {negative_salary}")

    validation_passed = (
        missing_cells == 0
        and duplicate_rows == 0
        and invalid_age == 0
        and negative_experience == 0
        and negative_salary == 0
    )

    if validation_passed:
        print("\nSTATUS: PASS")
        print("Cleaned dataset passed all quality checks.")

    else:
        print("\nSTATUS: FAIL")
        print("Cleaned dataset still contains data-quality issues.")

    return validation_passed


# ============================================================
# SAVE CLEANED DATA
# ============================================================

def save_cleaned_data(df):
    """
    Save cleaned data to data/processed/.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CLEANED_DATA_PATH,
        index=False
    )

    print(
        f"\nCleaned dataset saved to:\n"
        f"{CLEANED_DATA_PATH}"
    )


# ============================================================
# COMPLETE CLEANING PIPELINE
# ============================================================

def clean_employee_salary_data(df):
    """
    Execute the complete cleaning pipeline.
    """

    print("\n" + "=" * 70)
    print("EMPLOYEE SALARY — CLEANING PIPELINE")
    print("=" * 70)

    print(f"\nInitial rows: {len(df)}")

    # --------------------------------------------------------
    # 1. Remove invalid records
    # --------------------------------------------------------

    invalid_count = identify_invalid_records(df).sum()

    df = remove_invalid_records(df)

    print(
        f"After invalid-record removal: "
        f"{len(df)} rows "
        f"(-{invalid_count})"
    )

    # --------------------------------------------------------
    # 2. Remove duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    df = remove_duplicates(df)

    print(
        f"After duplicate removal: "
        f"{len(df)} rows "
        f"(-{duplicate_count})"
    )

    # --------------------------------------------------------
    # 3. Impute missing numerical values
    # --------------------------------------------------------

    missing_before = df.isna().sum().sum()

    df = impute_missing_values(df)

    missing_after = df.isna().sum().sum()

    print(
        f"Missing cells before imputation: "
        f"{missing_before}"
    )

    print(
        f"Missing cells after imputation : "
        f"{missing_after}"
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df_raw = load_raw_data()

    df_cleaned = clean_employee_salary_data(
        df_raw
    )

    validation_passed = validate_cleaned_data(
        df_cleaned
    )

    if validation_passed:

        save_cleaned_data(
            df_cleaned
        )

        print("\n" + "=" * 70)
        print("CLEANING COMPLETED SUCCESSFULLY")
        print("=" * 70)

    else:

        print("\n" + "=" * 70)
        print("CLEANING STOPPED — VALIDATION FAILED")
        print("=" * 70)