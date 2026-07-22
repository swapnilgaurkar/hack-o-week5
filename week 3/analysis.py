import pandas as pd
import numpy as np


class StudentAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.subjects = ["Math", "Science", "English", "Computer"]
        self.df = pd.read_csv(file_path)

    def clean_data(self):
        # Fill missing marks with the average of that subject
        for subject in self.subjects:
            self.df[subject] = self.df[subject].fillna(
                self.df[subject].mean()
            )

    def calculate_average(self):
        self.df["Average"] = self.df[self.subjects].mean(axis=1)

    def assign_grade(self):
        def grade(avg):
            if avg >= 90:
                return "A"
            elif avg >= 80:
                return "B"
            elif avg >= 70:
                return "C"
            elif avg >= 60:
                return "D"
            else:
                return "F"

        self.df["Grade"] = self.df["Average"].apply(grade)

    def top_students(self, n=5):
        return self.df.nlargest(n, "Average")

    def class_average(self):
        return self.df.groupby("Class")["Average"].mean()

    def subject_average(self):
        return {
            subject: round(self.df[subject].mean(), 2)
            for subject in self.subjects
        }

    def statistics(self):
        marks = self.df["Average"].to_numpy()

        return {
            "Mean": round(np.mean(marks), 2),
            "Median": round(np.median(marks), 2),
            "Standard Deviation": round(np.std(marks), 2),
            "25th Percentile": round(np.percentile(marks, 25), 2),
            "75th Percentile": round(np.percentile(marks, 75), 2)
        }

    def passed_students(self):
        return [
            name
            for name, avg in zip(self.df["Name"], self.df["Average"])
            if avg >= 40
        ]

    def save_results(self):
        self.df.to_csv("processed_students.csv", index=False)
