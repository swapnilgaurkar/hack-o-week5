import pandas as pd
import numpy as np

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    students = pd.read_csv("data/students.csv")
    branches = pd.read_csv("data/branch_details.csv")
    return students, branches


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df


# -----------------------------
# Merge DataFrames
# -----------------------------
def merge_data(students, branches):
    merged = pd.merge(students, branches, on="Branch")
    return merged


# -----------------------------
# Add Total, Average and Grade
# -----------------------------
def calculate_results(df):

    df["Total"] = df["Maths"] + df["Science"] + df["Python"]

    df["Average"] = (
        df["Maths"] +
        df["Science"] +
        df["Python"]
    ) / 3

    grades = []

    for avg in df["Average"]:
        if avg >= 90:
            grades.append("A+")
        elif avg >= 80:
            grades.append("A")
        elif avg >= 70:
            grades.append("B")
        elif avg >= 60:
            grades.append("C")
        elif avg >= 50:
            grades.append("D")
        else:
            grades.append("F")

    df["Grade"] = grades

    return df


# -----------------------------
# Top Performer
# -----------------------------
def topper(df):
    top = df.loc[df["Average"].idxmax()]
    print("\n========== TOPPER ==========")
    print(top[["Name", "Branch", "Average", "Grade"]])


# -----------------------------
# Lowest Performer
# -----------------------------
def lowest(df):
    low = df.loc[df["Average"].idxmin()]
    print("\n========== LOWEST SCORER ==========")
    print(low[["Name", "Branch", "Average", "Grade"]])


# -----------------------------
# Branch Wise Analysis
# -----------------------------
def branch_analysis(df):
    print("\n========== BRANCH ANALYSIS ==========\n")
    print(df.groupby("Branch")[["Maths", "Science", "Python", "Average"]].mean())


# -----------------------------
# Attendance Analysis
# -----------------------------
def attendance(df):

    print("\nAverage Attendance")

    print(df.groupby("Branch")["Attendance"].mean())


# -----------------------------
# Subject Statistics (NumPy)
# -----------------------------
def subject_statistics(df):

    maths = np.array(df["Maths"])
    science = np.array(df["Science"])
    python = np.array(df["Python"])

    print("\n========== SUBJECT STATISTICS ==========")

    print("Maths Average :", np.mean(maths))
    print("Science Average :", np.mean(science))
    print("Python Average :", np.mean(python))

    print("Highest Maths :", np.max(maths))
    print("Lowest Maths :", np.min(maths))


# -----------------------------
# Broadcasting
# -----------------------------
def bonus_marks(df):

    marks = df[["Maths", "Science", "Python"]].to_numpy()

    updated = marks + 5

    print("\nMarks after adding 5 bonus marks")

    print(updated)


# -----------------------------
# Vectorized Operation
# -----------------------------
def percentage(df):

    total = df["Maths"] + df["Science"] + df["Python"]

    df["Percentage"] = (total / 300) * 100

    print(df[["Name", "Percentage"]])


# -----------------------------
# List Comprehension
# -----------------------------
def passed_students(df):

    passed = [
        name
        for name, avg in zip(df["Name"], df["Average"])
        if avg >= 40
    ]

    print("\nPassed Students")

    print(passed)


# -----------------------------
# Dictionary Comprehension
# -----------------------------
def grades_dictionary(df):

    grade_dict = {
        row["Name"]: row["Grade"]
        for _, row in df.iterrows()
    }

    print("\nGrades Dictionary")

    print(grade_dict)