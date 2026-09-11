import numpy as np
import pandas as pd

K = 3
TRAIN_FILE_LOCATION = "DataSet"

def input_data() -> list:
    data = []



    return data



if __name__ == "__main__":



    distances = np.sqrt(np.sum((train_data[:, :3] - test_data) ** 2, axis = 1))
    # เรี ยงลาดับระยะห่าง
    indices = np.argsort(distances)
    # หาคลาสของ K เพื่อนบ้านที่ใกล้ที่สุด
    nearest_classes = train_data[indices[:K], 3]
    # นับจานวนคลาส
    class_counts = np.bincount(nearest_classes.astype(int))
    # หาคลาสที่มีจานวนมากที่สุด
    predicted_class = np.argmax(class_counts)
    # แสดงผลลัพธ์
    print("Test Data:")
    print(np.concatenate((test_data, [predicted_class])))

