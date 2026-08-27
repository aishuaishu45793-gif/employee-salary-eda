from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "03_employee_salary.csv"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load the original Employee Salary dataset."""
    return pd.read_csv(DATA_PATH)


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def check_missing_values(df):
    """Check missing values in every column."""

    print("\n[1] MISSING VALUES")
    print("-" * 50)

    missing = df.isnull().sum()

    print(missing)

    print(f"\nTotal missing cells: {missing.sum()}")


def check_duplicates(df):
    """Check duplicate rows."""

    print("\n[2] DUPLICATE ROWS")
    print("-" * 50)

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")


def check_negative_values(df):
    """Check negative values in numerical columns."""

    print("\n[3] NEGATIVE VALUES")
    print("-" * 50)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        count = (df[column] < 0).sum()

        print(f"{column:25}: {count}")


def check_zero_values(df):
    """Check zero values without automatically treating them as invalid."""

    print("\n[4] ZERO VALUES")
    print("-" * 50)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        count = (df[column] == 0).sum()

        print(f"{column:25}: {count}")


def check_expected_ranges(df):
    """
    Check values against reasonable ranges
    suggested by the structure of this dataset.

    These checks identify suspicious records.
    They do NOT modify the data.
    """

    print("\n[5] RANGE VALIDATION")
    print("-" * 50)

    # Experience
    invalid_experience = df[
        (df["experience_years"] < 0)
        | (df["experience_years"] > 20)
    ]

    print(
        f"experience_years outside 0-20 : "
        f"{len(invalid_experience)}"
    )

    # Age
    invalid_age = df[
        (df["age"] < 18)
        | (df["age"] > 60)
    ]

    print(
        f"age outside 18-60             : "
        f"{len(invalid_age)}"
    )

    # Projects
    invalid_projects = df[
        (df["projects_completed"] < 0)
        | (df["projects_completed"] > 20)
    ]

    print(
        f"projects outside 0-20        : "
        f"{len(invalid_projects)}"
    )

    # Training
    invalid_training = df[
        (df["training_hours"] < 0)
        | (df["training_hours"] > 100)
    ]

    print(
        f"training outside 0-100        : "
        f"{len(invalid_training)}"
    )

    # Salary
    invalid_salary = df[
        (df["salary_lakh"] <= 0)
        | (df["salary_lakh"] > 26.3)
    ]

    print(
        f"salary outside valid range   : "
        f"{len(invalid_salary)}"
    )


def show_suspicious_records(df):
    """
    Display records containing negative values.
    These records will be investigated before cleaning.
    """

    print("\n[6] SUSPICIOUS NEGATIVE RECORDS")
    print("-" * 50)

    negative_mask = (
        (df["experience_years"] < 0)
        | (df["salary_lakh"] < 0)
    )

    suspicious = df[negative_mask]

    if suspicious.empty:

        print("No negative experience/salary records found.")

    else:

        print(suspicious.to_string(index=True))


def generate_summary(df):
    """Generate a compact audit summary."""

    print("\n[7] FINAL AUDIT SUMMARY")
    print("-" * 50)

    print(f"Rows              : {len(df)}")
    print(f"Columns           : {len(df.columns)}")
    print(
        f"Missing cells     : "
        f"{df.isnull().sum().sum()}"
    )
    print(
        f"Duplicate rows    : "
        f"{df.duplicated().sum()}"
    )

    negative_count = 0

    for column in df.select_dtypes(
        include="number"
    ).columns:

        negative_count += (df[column] < 0).sum()

    print(f"Negative values   : {negative_count}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    print("\n" + "=" * 60)
    print("EMPLOYEE SALARY — ADVANCED DATA VALIDATION")
    print("=" * 60)

    print(f"\nDataset: {DATA_PATH}")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    check_missing_values(df)

    check_duplicates(df)

    check_negative_values(df)

    check_zero_values(df)

    check_expected_ranges(df)

    show_suspicious_records(df)

    generate_summary(df)

    print("\nValidation completed successfully.")