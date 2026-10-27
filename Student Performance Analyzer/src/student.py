class Student:
    def __init__(self, student_id, name, branch, gender, maths, science, python_marks, attendance):
        self.student_id = student_id
        self.name = name
        self.branch = branch
        self.gender = gender
        self.maths = int(maths)
        self.science = int(science)
        self.python_marks = int(python_marks)
        self.attendance = int(attendance)

    def total_marks(self):
        return self.maths + self.science + self.python_marks

    def average_marks(self):
        return round(self.total_marks() / 3, 2)

    def grade(self):
        avg = self.average_marks()

        if avg >= 90:
            return "A+"
        elif avg >= 80:
            return "A"
        elif avg >= 70:
            return "B"
        elif avg >= 60:
            return "C"
        elif avg >= 50:
            return "D"
        else:
            return "F"

    def is_pass(self):
        return (
            self.maths >= 35 and
            self.science >= 35 and
            self.python_marks >= 35
        )

    def display(self):
        print("-" * 60)
        print(f"Student ID : {self.student_id}")
        print(f"Name       : {self.name}")
        print(f"Branch     : {self.branch}")
        print(f"Gender     : {self.gender}")
        print(f"Maths      : {self.maths}")
        print(f"Science    : {self.science}")
        print(f"Python     : {self.python_marks}")
        print(f"Attendance : {self.attendance}%")
        print(f"Total      : {self.total_marks()}")
        print(f"Average    : {self.average_marks()}")
        print(f"Grade      : {self.grade()}")
        print(f"Result     : {'PASS' if self.is_pass() else 'FAIL'}")
        print("-" * 60)