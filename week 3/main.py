from analysis import StudentAnalyzer
from visualization import Visualizer

import os

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def header():
    clear_screen()
    print("=" * 60)
    print("      🎓 STUDENT PERFORMANCE ANALYSIS SYSTEM")
    print("=" * 60)
    print()


def menu():
    print("📌 MAIN MENU")
    print("-" * 60)
    print("1. 📋 View Student Data")
    print("2. 📊 Show Statistics")
    print("3. 📈 Visualizations")
    print("4. 🚪 Exit")
    print("-" * 60)

def print_header():
    print("\n" + "=" * 70)
    print("🎓      STUDENT PERFORMANCE ANALYSIS SYSTEM")
    print("=" * 70)
    print("📚 Student Report Dashboard")
    print("=" * 70)


def print_statistics(stats):
    print("\n📈 OVERALL STATISTICS")
    print("=" * 70)

    for key, value in stats.items():
        print(f"✅ {key:<25} : {value}")

    print("=" * 70)


def print_subject_average(subjects):
    print("\n📚 Subject Average")
    print("-" * 60)

    for subject, avg in subjects.items():
        print(f"{subject:<15}: {avg}")


def print_class_average(class_avg):
    print("\n🏫 Class Average")
    print("-" * 60)

    for cls, avg in class_avg.items():
        print(f"Class {cls:<10}: {round(avg,2)}")


def print_top_students(top_students):
    print("\n🏆 Top 5 Students")
    print("-" * 60)

    print(f"{'Rank':<5}{'Name':<15}{'Class':<10}{'Average'}")

    for i, (_, row) in enumerate(top_students.iterrows(), start=1):
        print(
            f"{i:<5}"
            f"{row['Name']:<15}"
            f"{row['Class']:<10}"
            f"{round(row['Average'],2)}"
        )


def main():

    analyzer = StudentAnalyzer("students.csv")

    analyzer.clean_data()
    analyzer.calculate_average()
    analyzer.assign_grade()

    print_header()

    print_statistics(analyzer.statistics())

    print_subject_average(analyzer.subject_average())

    print_class_average(analyzer.class_average())

    print_top_students(analyzer.top_students())

    analyzer.save_results()

    print("\n✅ Processed data saved as processed_students.csv")

    print("\n📊 Generating Charts...")

    visualizer = Visualizer(analyzer.df)

    visualizer.subject_average_chart()
    visualizer.grade_distribution()
    visualizer.top_students_chart()
    visualizer.correlation_heatmap()

    print("\n🎉 Analysis Completed Successfully!")


if __name__ == "__main__":
    main()
