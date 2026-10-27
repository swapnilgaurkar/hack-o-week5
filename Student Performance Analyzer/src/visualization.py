import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Bar Chart - Average Marks
# -----------------------------
def bar_chart(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Name"], df["Average"])
    plt.title("Student Average Marks")
    plt.xlabel("Students")
    plt.ylabel("Average Marks")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Histogram - Maths Marks
# -----------------------------
def histogram(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["Maths"], bins=5)
    plt.title("Distribution of Maths Marks")
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Pie Chart - Students by Branch
# -----------------------------
def pie_chart(df):
    branch_count = df["Branch"].value_counts()

    plt.figure(figsize=(7, 7))
    plt.pie(
        branch_count,
        labels=branch_count.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Students by Branch")
    plt.show()


# -----------------------------
# Box Plot - Subject Comparison
# -----------------------------
def box_plot(df):
    plt.figure(figsize=(8, 5))

    plt.boxplot([
        df["Maths"],
        df["Science"],
        df["Python"]
    ],
        labels=["Maths", "Science", "Python"]
    )

    plt.title("Marks Distribution")
    plt.ylabel("Marks")
    plt.show()


# -----------------------------
# Heatmap - Correlation Matrix
# -----------------------------
def heatmap(df):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df[["Maths", "Science", "Python", "Attendance", "Average"]].corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")
    plt.show()


# -----------------------------
# Subject Average Comparison
# -----------------------------
def subject_average(df):

    averages = [
        df["Maths"].mean(),
        df["Science"].mean(),
        df["Python"].mean()
    ]

    subjects = [
        "Maths",
        "Science",
        "Python"
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(subjects, averages)
    plt.title("Average Marks by Subject")
    plt.ylabel("Average")
    plt.show()


# -----------------------------
# Attendance Graph
# -----------------------------
def attendance_chart(df):
    plt.figure(figsize=(10, 5))

    plt.plot(
        df["Name"],
        df["Attendance"],
        marker="o"
    )

    plt.xticks(rotation=45)
    plt.title("Student Attendance")
    plt.xlabel("Students")
    plt.ylabel("Attendance (%)")
    plt.tight_layout()
    plt.show()