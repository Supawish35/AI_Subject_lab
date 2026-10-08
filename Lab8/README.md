# Lab 8: Artificial Neural Network (ANN) Classification & Reactive Web Simulator

โปรแกรมจำแนกและวินิจฉัยข้อมูลด้วยโครงข่ายประสาทเทียม (**Artificial Neural Network: ANN**) และเปรียบเทียบกับอัลกอริทึมที่เคยเรียนมาทั้งหมด ได้แก่:
1. **Artificial Neural Network (ANN / Multi-Layer Perceptron)** (Lab 8)
2. **Naive Bayes (GaussianNB)** (Lab 3)
3. **Decision Tree (CART)** (Lab 2)
4. **K-Nearest Neighbors (K-NN, K=5)** (Lab 4)

---

## 📌 ข้อกำหนดการทดลอง (Requirements)
- ทุกวิธีใช้สัดส่วนแบ่งข้อมูล: **Train 70%** และ **Test 30%** (Stratified Partitioning)
- วัดผลและเปรียบเทียบ 4 ตัวชี้วัดหลัก:
  - **Accuracy** (ความแม่นยำรวม)
  - **Precision** (ความเที่ยงตรง)
  - **Recall** (ความไว / ความครอบคลุม)
  - **F-Measure (F1-Score)** (ค่าเฉลี่ยฮาร์โมนิกของ Precision และ Recall)
- แสดง **Confusion Matrix (Actual Table)** ของทุกอัลกอริทึม
- หน้าเว็บจำลอง Reactive Web Simulator ออกแบบเป็น **Light Mode** สะอาดตาและตอบสนองทันใจ

---

## 📂 โครงสร้างไดเรกทอรี (Organized Project Architecture)

```
Lab8/
├── ann_model.py                     # Standalone ANN Classifier model (แยกโมเดลอิสระ)
├── main.py                          # สคริปต์หลักรันการทดลองและประเมินผล 4 อัลกอริทึม
├── server.py                        # Launcher สำหรับเปิดเซิร์ฟเวอร์ Reactive Web Simulator
├── README.md                        # เอกสารอธิบายการทดลองและวิธีใช้งาน
│
├── report/                          # 📁 ส่วนของรายงานผลการทดลอง (Report Part)
│   ├── generator.py                 # ตัวสร้าง HTML Dashboard สรุปผลการประเมิน
│   ├── lab8_report.html             # รายงานสรุปผลฉบับสมบูรณ์ (Interactive Dashboard)
│   ├── lab8_model_comparison.png    # กราฟแท่งเปรียบเทียบประสิทธิภาพทั้ง 4 อัลกอริทึม
│   └── lab8_confusion_matrices.png  # รูปภาพ Confusion Matrix Heatmaps ของทุกวิธี
│
└── reactive_form/                   # 📁 ส่วนของเว็บฟอร์มจำลอง (Reactive Form Part - Light Mode)
    ├── builder.py                   # ตัวสร้างหน้าเว็บจำลอง Light Theme พร้อมพารามิเตอร์โมเดล
    ├── server.py                    # เว็บเซิร์ฟเวอร์แบบ Real-Time และ REST API (/api/predict)
    └── index.html                   # หน้าเว็บ Reactive Simulator สไตล์ Light Mode
```

---

## 🚀 วิธีการรันโปรแกรม

### 1. รันการประเมินผลและเปรียบเทียบ 4 อัลกอริทึม (Train 70% / Test 30%)
```bash
python Lab8/main.py
```
*(เพิ่ม `--serve` เพื่อเปิดเว็บเซิร์ฟเวอร์ simulator ต่อท้ายการทดลองได้ทันที)*

### 2. รัน Reactive Web Simulator & ANN Server (Light Mode)
```bash
python Lab8/server.py
# หรือ
python Lab8/reactive_form/server.py 8080
```
เปิดเบราว์เซอร์ไปที่: **`http://localhost:8080/index.html`**

#### จุดเด่นของหน้าเว็บ Reactive Simulator (Light Mode):
1. **Light Theme Design:** ดีไซน์สว่าง สะอาดตา คอนทราสต์สูง อ่านง่าย สบายตา
2. **Interactive Feature Sliders:** ปรับเวลาอ่านหนังสือ, อัตราเข้าเรียน, เวลานอน, การใช้อินเทอร์เน็ต, การบ้านที่ส่ง, คะแนนสอบเก่า
3. **Interactive Neural Network Canvas:** แคนวาสจำลองสถาปัตยกรรมโครงข่ายประสาทเทียมสด ๆ (7 Inputs → 16 Hidden 1 → 8 Hidden 2 → 5 Outputs) พร้อมเส้น Synapses และเซลล์ประสาทที่เปลี่ยนสีตามระดับ Activation
4. **Sub-millisecond Reactivity:** คำนวณผลลัพธ์ทันทีที่ขยับ Slider (Client-Side JS Forward Pass + Live Server API)
5. **Instant Model Comparison:** แสดงผลการทำนายเปรียบเทียบระหว่าง ANN, Naive Bayes, Decision Tree และ K-NN ทันที
6. **Preset Personas:** ปุ่มลัดเลือกโหลดนักศึกษาเรียนดี (Grade A), ปานกลาง (Grade B/C), และกลุ่มเสี่ยง (Grade D/F)

---

## 📊 ผลการทดลองเปรียบเทียบ (Train 70% / Test 30%)

### ชุดข้อมูลหลัก: Student Performance Dataset (10,000 samples)
| Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F-Measure (%) |
| :--- | :---: | :---: | :---: | :---: |
| **ANN (Artificial Neural Network)** ⭐ | **76.27%** | **73.31%** | **76.27%** | **74.21%** |
| Naive Bayes (GaussianNB) | 73.70% | 66.27% | 73.70% | 67.58% |
| K-Nearest Neighbors (K=5) | 72.97% | 68.86% | 72.97% | 70.41% |
| Decision Tree | 66.97% | 67.95% | 66.97% | 67.44% |
