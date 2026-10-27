from analysis import *
from visualization import *

def main():

    # Load Data
    students, branches = load_data()

    # Clean Data
    students = clean_data(students)

    # Merge DataFrames
    merged = merge_data(students, branches)

    # Calculate Results
    merged = calculate_results(merged)

    while True:

        print("\n" + "=" * 55)
        print("      STUDENT PERFORMANCE ANALYZER")
        print("=" * 55)
        print("1. Display Student Records")
        print("2. Show Top Performer")
        print("3. Show Lowest Performer")
        print("4. Branch-wise Analysis")
        print("5. Attendance Analysis")
        print("6. Subject Statistics (NumPy)")
        print("7. Bonus Marks (Broadcasting)")
        print("8. Percentage Calculation (Vectorization)")
        print("9. Passed Students")
        print("10. Grades Dictionary")
        print("11. Show Graphs")
        print("12. Exit")
        print("=" * 55)

        choice = input("Enter your choice: ")

        if choice == "1":
            print("\nStudent Records\n")
            print(merged)

        elif choice == "2":
            topper(merged)

        elif choice == "3":
            lowest(merged)

        elif choice == "4":
            branch_analysis(merged)

        elif choice == "5":
            attendance(merged)

        elif choice == "6":
            subject_statistics(merged)

        elif choice == "7":
            bonus_marks(merged)

        elif choice == "8":
            percentage(merged)

        elif choice == "9":
            passed_students(merged)

        elif choice == "10":
            grades_dictionary(merged)

        elif choice == "11":

            print("\nGenerating Graphs...")

            bar_chart(merged)
            histogram(merged)
            pie_chart(merged)
            box_plot(merged)
            heatmap(merged)
            subject_average(merged)
            attendance_chart(merged)

        elif choice == "12":
            print("\nThank you for using Student Performance Analyzer!")
            break

        else:
            print("\nInvalid Choice! Please try again.")

if __name__ == "__main__":
    main()