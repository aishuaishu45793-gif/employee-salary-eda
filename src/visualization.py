from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "employee_salary_model_ready.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the model-ready employee salary dataset."""
    return pd.read_csv(DATA_PATH)


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():
    """Create reports/figures if it does not exist."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 1. SALARY DISTRIBUTION
# ============================================================

def plot_salary_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["salary_lakh"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Salary Distribution")
    plt.xlabel("Salary (Lakh)")
    plt.ylabel("Number of Employees")

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "salary_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Created: {output_path}")


# ============================================================
# 2. EXPERIENCE DISTRIBUTION
# ============================================================

def plot_experience_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["experience_years"],
        bins=20,
        edgecolor="black"
    )

    plt.title("Experience Distribution")
    plt.xlabel("Experience (Years)")
    plt.ylabel("Number of Employees")

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "experience_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Created: {output_path}")


# ============================================================
# 3. EXPERIENCE VS SALARY
# ============================================================

def plot_experience_vs_salary(df):

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["experience_years"],
        df["salary_lakh"],
        alpha=0.6
    )

    plt.title(
        "Experience vs Salary"
    )

    plt.xlabel(
        "Experience (Years)"
    )

    plt.ylabel(
        "Salary (Lakh)"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "experience_vs_salary.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Created: {output_path}")


# ============================================================
# 4. PROJECTS VS SALARY
# ============================================================

def plot_projects_vs_salary(df):

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["projects_completed"],
        df["salary_lakh"],
        alpha=0.6
    )

    plt.title(
        "Projects Completed vs Salary"
    )

    plt.xlabel(
        "Projects Completed"
    )

    plt.ylabel(
        "Salary (Lakh)"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "projects_vs_salary.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Created: {output_path}")


# ============================================================
# 5. TRAINING VS SALARY
# ============================================================

def plot_training_vs_salary(df):

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["training_hours"],
        df["salary_lakh"],
        alpha=0.6
    )

    plt.title(
        "Training Hours vs Salary"
    )

    plt.xlabel(
        "Training Hours"
    )

    plt.ylabel(
        "Salary (Lakh)"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "training_vs_salary.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Created: {output_path}")


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    numeric_columns = [
        "experience_years",
        "age",
        "projects_completed",
        "training_hours",
        "experience_score",
        "project_experience_ratio",
        "salary_lakh"
    ]

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    plt.figure(
        figsize=(11, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title(
        "Employee Salary Correlation Heatmap"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "correlation_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Created: {output_path}")


# ============================================================
# MAIN VISUALIZATION PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("EMPLOYEE SALARY — VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nLoaded dataset: "
        f"{len(df)} rows × {len(df.columns)} columns"
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    prepare_output_directory()

    # --------------------------------------------------------
    # Create visualizations
    # --------------------------------------------------------

    plot_salary_distribution(df)

    plot_experience_distribution(df)

    plot_experience_vs_salary(df)

    plot_projects_vs_salary(df)

    plot_training_vs_salary(df)

    plot_correlation_heatmap(df)

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nFigures saved in:\n{FIGURES_DIR}"
    )