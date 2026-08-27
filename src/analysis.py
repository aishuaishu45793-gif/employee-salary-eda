from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_READY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "employee_salary_model_ready.csv"
)


# ============================================================
# LOAD MODEL-READY DATA
# ============================================================

def load_data():
    """
    Load the cleaned and feature-engineered dataset.
    """

    return pd.read_csv(
        MODEL_READY_PATH
    )


# ============================================================
# BASIC SALARY STATISTICS
# ============================================================

def salary_statistics(df):

    salary = df["salary_lakh"]

    statistics = {
        "count": salary.count(),
        "mean": salary.mean(),
        "median": salary.median(),
        "std": salary.std(),
        "minimum": salary.min(),
        "maximum": salary.max(),
        "q1": salary.quantile(0.25),
        "q3": salary.quantile(0.75)
    }

    return statistics


# ============================================================
# EXPERIENCE ANALYSIS
# ============================================================

def experience_analysis(df):

    return (
        df.groupby("experience_years")["salary_lakh"]
        .agg(
            employees="count",
            average_salary="mean",
            median_salary="median"
        )
        .reset_index()
        .sort_values("experience_years")
    )


# ============================================================
# PROJECT ANALYSIS
# ============================================================

def project_analysis(df):

    return (
        df.groupby("projects_completed")["salary_lakh"]
        .agg(
            employees="count",
            average_salary="mean",
            median_salary="median"
        )
        .reset_index()
        .sort_values("projects_completed")
    )


# ============================================================
# TRAINING ANALYSIS
# ============================================================

def training_analysis(df):

    df = df.copy()

    # Create practical training groups
    df["training_group"] = pd.cut(
        df["training_hours"],
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=[
            "0-25 hours",
            "26-50 hours",
            "51-75 hours",
            "76+ hours"
        ]
    )

    return (
        df.groupby(
            "training_group",
            observed=False
        )["salary_lakh"]
        .agg(
            employees="count",
            average_salary="mean",
            median_salary="median"
        )
        .reset_index()
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

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

    salary_correlations = (
        correlation_matrix["salary_lakh"]
        .drop("salary_lakh")
        .sort_values(
            ascending=False
        )
    )

    return correlation_matrix, salary_correlations


# ============================================================
# TOP RELATIONSHIPS
# ============================================================

def strongest_relationships(df):

    correlation_matrix = (
        df.select_dtypes(
            include=np.number
        )
        .corr()
    )

    pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            feature_1 = columns[i]
            feature_2 = columns[j]

            correlation = correlation_matrix.loc[
                feature_1,
                feature_2
            ]

            pairs.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "absolute_correlation": abs(
                        correlation
                    )
                }
            )

    relationships = pd.DataFrame(
        pairs
    )

    return relationships.sort_values(
        "absolute_correlation",
        ascending=False
    )


# ============================================================
# SALARY GROUPS
# ============================================================

def salary_groups(df):

    df = df.copy()

    df["salary_group"] = pd.qcut(
        df["salary_lakh"],
        q=4,
        labels=[
            "Lower Salary",
            "Lower-Middle Salary",
            "Upper-Middle Salary",
            "Higher Salary"
        ],
        duplicates="drop"
    )

    return (
        df.groupby(
            "salary_group",
            observed=False
        )["salary_lakh"]
        .agg(
            employees="count",
            average_salary="mean",
            median_salary="median"
        )
        .reset_index()
    )


# ============================================================
# EXPERIENCE GROUPS
# ============================================================

def experience_groups(df):

    df = df.copy()

    df["experience_group"] = pd.cut(
        df["experience_years"],
        bins=[
            -np.inf,
            2,
            5,
            10,
            15,
            np.inf
        ],
        labels=[
            "0-2 years",
            "3-5 years",
            "6-10 years",
            "11-15 years",
            "16+ years"
        ]
    )

    return (
        df.groupby(
            "experience_group",
            observed=False
        )["salary_lakh"]
        .agg(
            employees="count",
            average_salary="mean",
            median_salary="median"
        )
        .reset_index()
    )


# ============================================================
# KEY INSIGHTS
# ============================================================

def generate_insights(
    df,
    salary_correlations
):

    insights = []

    # --------------------------------------------------------
    # Salary
    # --------------------------------------------------------

    salary = df["salary_lakh"]

    insights.append(
        f"The average salary is "
        f"{salary.mean():.2f} lakh, while the median salary is "
        f"{salary.median():.2f} lakh."
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    strongest_feature = (
        salary_correlations.index[0]
    )

    strongest_value = (
        salary_correlations.iloc[0]
    )

    insights.append(
        f"The strongest positive linear relationship "
        f"with salary is {strongest_feature}, "
        f"with a correlation of "
        f"{strongest_value:.3f}."
    )

    # --------------------------------------------------------
    # Highest salary
    # --------------------------------------------------------

    highest_salary_row = df.loc[
        df["salary_lakh"].idxmax()
    ]

    insights.append(
        f"The highest observed salary is "
        f"{highest_salary_row['salary_lakh']:.2f} lakh."
    )

    # --------------------------------------------------------
    # Lowest salary
    # --------------------------------------------------------

    lowest_salary_row = df.loc[
        df["salary_lakh"].idxmin()
    ]

    insights.append(
        f"The lowest observed salary is "
        f"{lowest_salary_row['salary_lakh']:.2f} lakh."
    )

    # --------------------------------------------------------
    # Experience difference
    # --------------------------------------------------------

    correlation = df[
        "experience_years"
    ].corr(
        df["salary_lakh"]
    )

    insights.append(
        f"Experience years have a salary correlation "
        f"of {correlation:.3f}."
    )

    return insights


# ============================================================
# PRINT REPORT
# ============================================================

def print_analysis_report(
    df,
    statistics,
    salary_correlations,
    insights
):

    print("\n" + "=" * 70)
    print("EMPLOYEE SALARY — ANALYTICAL REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nDATASET")
    print("-" * 70)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Salary statistics
    # --------------------------------------------------------

    print("\nSALARY STATISTICS")
    print("-" * 70)

    for key, value in statistics.items():

        if key == "count":
            print(
                f"{key:15}: {int(value)}"
            )
        else:
            print(
                f"{key:15}: {value:.3f}"
            )

    # --------------------------------------------------------
    # Salary correlations
    # --------------------------------------------------------

    print("\nCORRELATION WITH SALARY")
    print("-" * 70)

    print(
        salary_correlations.to_string()
    )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    print("\nKEY INSIGHTS")
    print("-" * 70)

    for number, insight in enumerate(
        insights,
        start=1
    ):

        print(
            f"{number}. {insight}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("EMPLOYEE SALARY — DATA ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nLoaded model-ready dataset: "
        f"{len(df)} rows"
    )

    # --------------------------------------------------------
    # Salary statistics
    # --------------------------------------------------------

    statistics = salary_statistics(
        df
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    (
        correlation_matrix,
        salary_correlations
    ) = correlation_analysis(
        df
    )

    # --------------------------------------------------------
    # Generate insights
    # --------------------------------------------------------

    insights = generate_insights(
        df,
        salary_correlations
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_analysis_report(
        df,
        statistics,
        salary_correlations,
        insights
    )

    # --------------------------------------------------------
    # Additional analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIENCE ANALYSIS")
    print("=" * 70)

    print(
        experience_analysis(df)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("PROJECT ANALYSIS")
    print("=" * 70)

    print(
        project_analysis(df)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("TRAINING ANALYSIS")
    print("=" * 70)

    print(
        training_analysis(df)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("SALARY GROUP ANALYSIS")
    print("=" * 70)

    print(
        salary_groups(df)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("EXPERIENCE GROUP ANALYSIS")
    print("=" * 70)

    print(
        experience_groups(df)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STRONGEST FEATURE RELATIONSHIPS")
    print("=" * 70)

    print(
        strongest_relationships(df)
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)