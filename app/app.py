from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "employee_salary_model_ready.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Employee & HR Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def salary_group(value, q1, q3):

    if value <= q1:
        return "Lower Salary"

    elif value <= q3:
        return "Middle Salary"

    return "Higher Salary"


def create_salary_groups(data):

    data = data.copy()

    q1 = data["salary_lakh"].quantile(0.25)
    q3 = data["salary_lakh"].quantile(0.75)

    data["salary_group"] = data[
        "salary_lakh"
    ].apply(
        lambda x: salary_group(x, q1, q3)
    )

    return data


def create_experience_groups(data):

    data = data.copy()

    data["experience_group"] = pd.cut(
        data["experience_years"],
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

    return data


def create_age_groups(data):

    data = data.copy()

    data["age_group"] = pd.cut(
        data["age"],
        bins=[
            17,
            25,
            35,
            45,
            55,
            np.inf
        ],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56+"
        ]
    )

    return data


def create_project_groups(data):

    data = data.copy()

    data["project_group"] = pd.cut(
        data["projects_completed"],
        bins=[
            -np.inf,
            5,
            10,
            15,
            np.inf
        ],
        labels=[
            "0-5",
            "6-10",
            "11-15",
            "16+"
        ]
    )

    return data


def create_training_groups(data):

    data = data.copy()

    data["training_group"] = pd.cut(
        data["training_hours"],
        bins=[
            -np.inf,
            25,
            50,
            75,
            np.inf
        ],
        labels=[
            "0-25 hours",
            "26-50 hours",
            "51-75 hours",
            "76+ hours"
        ]
    )

    return data


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💼 HR Analytics")

st.sidebar.markdown(
    "### Dashboard Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Overview",
        "Employee Analytics",
        "Compensation",
        "Experience & Career",
        "Projects & Productivity",
        "Learning & Training",
        "Advanced Analytics",
        "Employee Explorer",
        "HR Metrics Roadmap"
    ]
)


# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

experience_range = st.sidebar.slider(
    "Experience (Years)",
    float(df["experience_years"].min()),
    float(df["experience_years"].max()),
    (
        float(df["experience_years"].min()),
        float(df["experience_years"].max())
    )
)

salary_range = st.sidebar.slider(
    "Salary (Lakh)",
    float(df["salary_lakh"].min()),
    float(df["salary_lakh"].max()),
    (
        float(df["salary_lakh"].min()),
        float(df["salary_lakh"].max())
    )
)

age_range = st.sidebar.slider(
    "Age",
    int(df["age"].min()),
    int(df["age"].max()),
    (
        int(df["age"].min()),
        int(df["age"].max())
    )
)

project_range = st.sidebar.slider(
    "Projects Completed",
    int(df["projects_completed"].min()),
    int(df["projects_completed"].max()),
    (
        int(df["projects_completed"].min()),
        int(df["projects_completed"].max())
    )
)

training_range = st.sidebar.slider(
    "Training Hours",
    int(df["training_hours"].min()),
    int(df["training_hours"].max()),
    (
        int(df["training_hours"].min()),
        int(df["training_hours"].max())
    )
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[
    (df["experience_years"] >= experience_range[0])
    & (df["experience_years"] <= experience_range[1])
    & (df["salary_lakh"] >= salary_range[0])
    & (df["salary_lakh"] <= salary_range[1])
    & (df["age"] >= age_range[0])
    & (df["age"] <= age_range[1])
    & (df["projects_completed"] >= project_range[0])
    & (df["projects_completed"] <= project_range[1])
    & (df["training_hours"] >= training_range[0])
    & (df["training_hours"] <= training_range[1])
].copy()


# ============================================================
# HEADER
# ============================================================

st.title("💼 Employee & HR Analytics Dashboard")

st.markdown(
    """
    **Employee Salary EDA + Workforce Analytics**

    Analyze compensation, experience, projects, training,
    employee segmentation and workforce patterns.
    """
)

st.info(
    "Current dataset supports employee salary, experience, "
    "age, projects and training analytics. HR metrics such "
    "as attrition, attendance and performance require "
    "additional HR data."
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("📊 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Employees",
            f"{len(filtered_df):,}"
        )

    with col2:
        st.metric(
            "Average Salary",
            f"{filtered_df['salary_lakh'].mean():.2f} L"
        )

    with col3:
        st.metric(
            "Median Salary",
            f"{filtered_df['salary_lakh'].median():.2f} L"
        )

    with col4:
        st.metric(
            "Average Experience",
            f"{filtered_df['experience_years'].mean():.2f} yrs"
        )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Age",
            f"{filtered_df['age'].mean():.1f}"
        )

    with col2:
        st.metric(
            "Average Projects",
            f"{filtered_df['projects_completed'].mean():.1f}"
        )

    with col3:
        st.metric(
            "Training / Employee",
            f"{filtered_df['training_hours'].mean():.1f} hrs"
        )

    with col4:
        st.metric(
            "Maximum Salary",
            f"{filtered_df['salary_lakh'].max():.2f} L"
        )

    st.subheader("💰 Salary Distribution")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        filtered_df["salary_lakh"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Salary (Lakh)")
    ax.set_ylabel("Employees")
    ax.set_title("Salary Distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📈 Experience vs Salary")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        filtered_df["experience_years"],
        filtered_df["salary_lakh"],
        alpha=0.6
    )

    ax.set_xlabel("Experience (Years)")
    ax.set_ylabel("Salary (Lakh)")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# EMPLOYEE ANALYTICS
# ============================================================

elif page == "Employee Analytics":

    st.header("👨‍💼 Employee Analytics")

    data = create_age_groups(
        filtered_df
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Age Distribution")

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            data["age"],
            bins=20,
            edgecolor="black"
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Employees")

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.subheader("Employees by Age Group")

        age_summary = (
            data
            .groupby(
                "age_group",
                observed=False
            )
            .size()
        )

        st.bar_chart(
            age_summary
        )

    st.subheader("Employee Statistics")

    stats = pd.DataFrame({
        "Metric": [
            "Employees",
            "Average Age",
            "Minimum Age",
            "Maximum Age",
            "Average Experience",
            "Average Projects",
            "Average Training Hours"
        ],
        "Value": [
            len(data),
            round(data["age"].mean(), 2),
            data["age"].min(),
            data["age"].max(),
            round(data["experience_years"].mean(), 2),
            round(data["projects_completed"].mean(), 2),
            round(data["training_hours"].mean(), 2)
        ]
    })

    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# COMPENSATION
# ============================================================

elif page == "Compensation":

    st.header("💰 Compensation Analytics")

    salary = filtered_df["salary_lakh"]

    q1 = salary.quantile(0.25)
    q2 = salary.quantile(0.50)
    q3 = salary.quantile(0.75)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Minimum",
            f"{salary.min():.2f} L"
        )

    with col2:
        st.metric(
            "Q1",
            f"{q1:.2f} L"
        )

    with col3:
        st.metric(
            "Median",
            f"{q2:.2f} L"
        )

    with col4:
        st.metric(
            "Q3",
            f"{q3:.2f} L"
        )

    st.subheader("Salary Groups")

    data = create_salary_groups(
        filtered_df
    )

    salary_summary = (
        data
        .groupby(
            "salary_group",
            observed=False
        )["salary_lakh"]
        .agg(
            Employees="count",
            Average="mean",
            Median="median"
        )
        .reset_index()
    )

    st.dataframe(
        salary_summary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Salary vs Experience")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.regplot(
        data=filtered_df,
        x="experience_years",
        y="salary_lakh",
        scatter_kws={"alpha": 0.5},
        ax=ax
    )

    ax.set_xlabel("Experience (Years)")
    ax.set_ylabel("Salary (Lakh)")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Salary vs Projects")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        filtered_df["projects_completed"],
        filtered_df["salary_lakh"],
        alpha=0.6
    )

    ax.set_xlabel("Projects")
    ax.set_ylabel("Salary (Lakh)")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# EXPERIENCE & CAREER
# ============================================================

elif page == "Experience & Career":

    st.header("📈 Experience & Career Analytics")

    data = create_experience_groups(
        filtered_df
    )

    summary = (
        data
        .groupby(
            "experience_group",
            observed=False
        )["salary_lakh"]
        .agg(
            Employees="count",
            Average_Salary="mean",
            Median_Salary="median"
        )
        .reset_index()
    )

    st.subheader(
        "Average Salary by Experience Group"
    )

    st.bar_chart(
        summary.set_index(
            "experience_group"
        )["Average_Salary"]
    )

    st.subheader(
        "Experience Group Summary"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    correlation = filtered_df[
        "experience_years"
    ].corr(
        filtered_df["salary_lakh"]
    )

    st.metric(
        "Experience-Salary Correlation",
        f"{correlation:.3f}"
    )

    st.subheader(
        "Experience Score Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        filtered_df["experience_score"],
        bins=20,
        edgecolor="black"
    )

    ax.set_xlabel("Experience Score")
    ax.set_ylabel("Employees")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# PROJECTS & PRODUCTIVITY
# ============================================================

elif page == "Projects & Productivity":

    st.header("📁 Projects & Productivity")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Projects",
            f"{filtered_df['projects_completed'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Maximum Projects",
            int(
                filtered_df[
                    "projects_completed"
                ].max()
            )
        )

    with col3:
        st.metric(
            "Project-Salary Correlation",
            f"{filtered_df['projects_completed'].corr(filtered_df['salary_lakh']):.3f}"
        )

    data = create_project_groups(
        filtered_df
    )

    summary = (
        data
        .groupby(
            "project_group",
            observed=False
        )["salary_lakh"]
        .agg(
            Employees="count",
            Average_Salary="mean"
        )
        .reset_index()
    )

    st.subheader(
        "Salary by Project Group"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        filtered_df["projects_completed"],
        filtered_df["salary_lakh"],
        alpha=0.6
    )

    ax.set_xlabel("Projects Completed")
    ax.set_ylabel("Salary (Lakh)")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# TRAINING
# ============================================================

elif page == "Learning & Training":

    st.header("🎓 Learning & Development")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Training",
            f"{filtered_df['training_hours'].mean():.2f} hrs"
        )

    with col2:
        st.metric(
            "Maximum Training",
            f"{filtered_df['training_hours'].max():.0f} hrs"
        )

    with col3:
        st.metric(
            "Training-Salary Correlation",
            f"{filtered_df['training_hours'].corr(filtered_df['salary_lakh']):.3f}"
        )

    data = create_training_groups(
        filtered_df
    )

    summary = (
        data
        .groupby(
            "training_group",
            observed=False
        )["salary_lakh"]
        .agg(
            Employees="count",
            Average_Salary="mean"
        )
        .reset_index()
    )

    st.subheader(
        "Salary by Training Group"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Training Hours Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        filtered_df["training_hours"],
        bins=20,
        edgecolor="black"
    )

    ax.set_xlabel("Training Hours")
    ax.set_ylabel("Employees")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# ADVANCED ANALYTICS
# ============================================================

elif page == "Advanced Analytics":

    st.header("🧠 Advanced Analytics")

    st.subheader(
        "Feature Correlation Matrix"
    )

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
        filtered_df[numeric_columns]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader(
        "Features Most Associated with Salary"
    )

    salary_corr = (
        correlation_matrix[
            "salary_lakh"
        ]
        .drop("salary_lakh")
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        salary_corr
    )

    st.subheader(
        "Salary Outlier Detection"
    )

    q1 = filtered_df[
        "salary_lakh"
    ].quantile(0.25)

    q3 = filtered_df[
        "salary_lakh"
    ].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = filtered_df[
        (filtered_df["salary_lakh"] < lower)
        | (filtered_df["salary_lakh"] > upper)
    ]

    st.metric(
        "Salary Outliers",
        len(outliers)
    )

    st.dataframe(
        outliers,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EMPLOYEE EXPLORER
# ============================================================

elif page == "Employee Explorer":

    st.header("🔎 Employee Explorer")

    display_columns = [
        "experience_years",
        "age",
        "projects_completed",
        "training_hours",
        "experience_score",
        "project_experience_ratio",
        "salary_lakh"
    ]

    explorer_df = filtered_df[
        display_columns
    ].copy()

    explorer_df = create_salary_groups(
        explorer_df
    )

    st.write(
        f"Showing {len(explorer_df):,} employees"
    )

    st.dataframe(
        explorer_df,
        use_container_width=True,
        hide_index=True
    )

    csv = explorer_df.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Filtered Employee Data",
        data=csv,
        file_name="filtered_employee_data.csv",
        mime="text/csv"
    )


# ============================================================
# HR METRICS ROADMAP
# ============================================================

elif page == "HR Metrics Roadmap":

    st.header("🏢 Company HR Analytics Roadmap")

    st.markdown(
        """
        A production HR analytics system normally combines
        employee, payroll, recruitment, attendance, learning,
        performance and engagement data.
        """
    )

    metrics = pd.DataFrame({
        "HR Metric": [
            "Headcount",
            "Average Salary",
            "Salary Distribution",
            "Experience",
            "Training Hours",
            "Projects Completed",
            "Turnover Rate",
            "Attrition Rate",
            "Retention Rate",
            "Absenteeism Rate",
            "Attendance Rate",
            "Performance Rating",
            "Employee Engagement",
            "Time to Hire",
            "Time to Fill",
            "Offer Acceptance Rate",
            "Cost per Hire",
            "Promotion Rate",
            "Internal Mobility",
            "Training Completion",
            "Training ROI",
            "Payroll Cost",
            "Benefits Cost",
            "Revenue per Employee",
            "Pay Equity"
        ],
        "Current Dataset": [
            "Available",
            "Available",
            "Available",
            "Available",
            "Available",
            "Available",
            "Requires HR data",
            "Requires HR data",
            "Requires HR data",
            "Requires attendance data",
            "Requires attendance data",
            "Requires performance data",
            "Requires survey data",
            "Requires recruitment data",
            "Requires recruitment data",
            "Requires recruitment data",
            "Requires recruitment data",
            "Requires promotion data",
            "Requires employee history",
            "Requires LMS data",
            "Requires training cost data",
            "Requires payroll data",
            "Requires benefits data",
            "Requires revenue data",
            "Requires demographic/pay data"
        ]
    })

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These additional metrics should be calculated "
        "from real company HR systems rather than invented "
        "from the current salary dataset."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Employee & HR Analytics Dashboard | "
    "Python • Pandas • Matplotlib • Seaborn • Streamlit"
)