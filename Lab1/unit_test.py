from normalizeData import  find_average, find_standard_deviation

FILE_PATH: str = "archive/student_dataset_10000_rows.csv"

if __name__ == "__main__":
    print(find_average(FILE_PATH, "study_hours"))
    print(find_standard_deviation(FILE_PATH, "study_hours"))
