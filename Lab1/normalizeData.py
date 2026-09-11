import pandas as pd
import math

CHUNK_SIZE: int = 500
FILE_PATH: str = "archive/student_dataset_10000_rows.csv"
OUTPUT_FILE: str = "student_dataset_10000_rows_normalized.csv"

#? find average
def find_average(file_path: str, column_name: str) -> float:
    total_sum: float = 0.0
    total_count: int = 0

    #? Ensure we only load the column we need to save memory
    for chunk in pd.read_csv(file_path, chunksize = CHUNK_SIZE, usecols = [column_name]):
        #? Target the specific column series to get a float, not a pandas Series object
        total_sum += chunk[column_name].sum()
        total_count += chunk[column_name].count()

    return float(total_sum / total_count)


def find_standard_deviation(file_path: str, column_name: str) -> float:
    squared_diff_sum: float = 0.0
    mean: float = find_average(file_path, column_name)
    total_count: int = 0

    for chunk in pd.read_csv(file_path, chunksize = CHUNK_SIZE, usecols = [column_name]):
        #? Calculate sum of squared differences for this chunk
        squared_diff_sum += ((chunk[column_name] - mean) ** 2).sum()
        total_count += chunk[column_name].count()

    variance = squared_diff_sum / total_count  #? Using population variance (divide by N)
    return math.sqrt(variance)


def normalize_fields(fields_names: list, file_path: str):
    all_average: dict = {}
    all_std: dict = {}
    is_first_chunk: bool = True

    print("Step 1: Calculating statistics...")
    #? Calculate average and standard deviation for all requested fields first
    for field in fields_names:
        # * Calculate the mean and std
        all_average[field] = find_average(file_path, field)
        all_std[field] = find_standard_deviation(file_path, field)

        print(f" - {field}: Mean = {all_average[field]:.2f}, Std = {all_std[field]:.2f}")

    print("Step 2: Normalizing and writing to new file...")
    #? Iterate through the file one last time to apply the math and write it out
    for chunk in pd.read_csv(file_path, chunksize = CHUNK_SIZE):

        for field in fields_names:
            #? Standard Z-Score Normalization: (Value - Mean) / Standard Deviation
            chunk[field] = (chunk[field] - all_average[field]) / all_std[field]

        #? Append the processed chunk to the new file
        chunk.to_csv(OUTPUT_FILE, mode = 'a', index = False, header = is_first_chunk)
        is_first_chunk = False


if __name__ == "__main__":
    #? Replace these with the actual numeric columns in your CSV you want to normalize
    fields_to_normalize = ["study_hours", "attendance", "sleep_hours", "internet_usage", "assignments_completed", "previous_score", "exam_score"]

    normalize_fields(fields_to_normalize, FILE_PATH)
    print(f"Success! Normalized data saved to {OUTPUT_FILE}")