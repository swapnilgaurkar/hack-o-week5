import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11
plt.rcParams["figure.facecolor"] = "white"
import os


class Visualizer:
    def __init__(self, dataframe):
        self.df = dataframe
        self.subjects = ["Math", "Science", "English", "Computer"]

        # Create output folder if it doesn't exist
        os.makedirs("output", exist_ok=True)

    def subject_average_chart(self):
        averages = self.df[self.subjects].mean()

        plt.figure(figsize=(10, 6))

        bars = plt.bar(
            averages.index,
            averages.values,
            color="royalblue",
            edgecolor="black",
            linewidth=1.5
        )

        # Display values on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.2,
                f"{height:.1f}",
                ha="center",
                fontsize=10,
                fontweight="bold"
            )

        plt.title("📊 Subject-wise Average Marks", fontsize=16, fontweight="bold")
        plt.xlabel("Subjects", fontsize=12)
        plt.ylabel("Average Marks", fontsize=12)

        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        plt.savefig("output/subject_average.png")
        plt.show()

    def grade_distribution(self):
        plt.figure(figsize=(6, 4))

        sns.countplot(data=self.df, x="Grade", palette="Set2")

        plt.title("Grade Distribution")
        plt.xlabel("Grade")
        plt.ylabel("Number of Students")

        plt.tight_layout()
        plt.savefig("output/grade_distribution.png")
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    def top_students_chart(self):
        top = self.df.nlargest(5, "Average")

        plt.figure(figsize=(8, 5))

        sns.barplot(
            data=top,
            x="Average",
            y="Name",
            palette="viridis"
        )

        plt.title("Top 5 Students")
        plt.xlabel("Average Marks")
        plt.ylabel("Student")

        plt.tight_layout()
        plt.savefig("output/top_students.png")
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self):
        plt.figure(figsize=(6, 5))

        sns.heatmap(
            self.df[self.subjects].corr(),
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Subject Correlation")

        plt.tight_layout()
        plt.savefig("output/correlation_heatmap.png")
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

