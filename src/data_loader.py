from pathlib import Path
import pandas as pd


# Get the root folder of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the actual Employee Salary dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "03_employee_salary.csv"


def load_employee_salary_data():
    """
    Load the Employee Salary dataset from the workbook.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


if __name__ == "__main__":

    df = load_employee_salary_data()

    print("\n========== DATASET LOADED SUCCESSFULLY ==========")

    print(f"\nDataset path: {DATA_PATH}")

    print(f"Rows: {df.shape[0]}")

    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)