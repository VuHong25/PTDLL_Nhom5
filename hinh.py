
# 1. IMPORT THƯ VIỆN
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# =========================
# 2. ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH
# =========================
df = pd.read_csv("output/student_clean.csv")

print(df.head())

# =========================
# 3. PHÂN TÍCH ĐƠN BIẾN
# =========================

# 3.1 Histogram – StudyHours
plt.figure(figsize=(8, 5))
sns.histplot(df["StudyHours"], kde=True)
plt.title("Phân phối thời gian học (StudyHours)", fontweight="bold")
plt.xlabel("StudyHours")
plt.ylabel("Tần suất")
plt.tight_layout()
plt.show()

# 3.2 Histogram – Attendance
plt.figure(figsize=(8, 5))
sns.histplot(df["Attendance"], kde=True)
plt.title("Phân phối tỷ lệ chuyên cần (Attendance)", fontweight="bold")
plt.xlabel("Attendance (%)")
plt.ylabel("Tần suất")
plt.tight_layout()
plt.show()

# 3.3 Histogram – ExamScore
plt.figure(figsize=(8, 5))
sns.histplot(df["ExamScore"], kde=True)
plt.title("Phân phối điểm thi (ExamScore)", fontweight="bold")
plt.xlabel("ExamScore")
plt.ylabel("Tần suất")
plt.tight_layout()
plt.show()

# 3.4 Bar chart – FinalGrade
plt.figure(figsize=(7, 5))
grade_counts = df["FinalGrade"].value_counts().sort_index()
plt.bar(grade_counts.index, grade_counts.values, edgecolor="black")
plt.title("Phân phối điểm cuối kỳ (FinalGrade)", fontweight="bold")
plt.xlabel("FinalGrade")
plt.ylabel("Số lượng sinh viên")
plt.xticks(grade_counts.index)
plt.tight_layout()
plt.show()

# =========================
# 4. PHÂN TÍCH ĐA BIẾN
# =========================

# 4.1 Scatter – StudyHours vs FinalGrade
plt.figure(figsize=(8, 5))
sns.scatterplot(x="StudyHours", y="FinalGrade", data=df, alpha=0.6)
plt.title("StudyHours vs FinalGrade", fontweight="bold")
plt.xlabel("StudyHours")
plt.ylabel("FinalGrade")
plt.tight_layout()
plt.show()

# 4.2 Scatter – Attendance vs FinalGrade
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Attendance", y="FinalGrade", data=df, alpha=0.6)
plt.title("Attendance vs FinalGrade", fontweight="bold")
plt.xlabel("Attendance")
plt.ylabel("FinalGrade")
plt.tight_layout()
plt.show()

# 4.3 Scatter – ExamScore vs FinalGrade
plt.figure(figsize=(8, 5))
sns.scatterplot(x="ExamScore", y="FinalGrade", data=df, alpha=0.6)
plt.title("ExamScore vs FinalGrade", fontweight="bold")
plt.xlabel("ExamScore")
plt.ylabel("FinalGrade")
plt.tight_layout()
plt.show()

# 4.4 Scatter – StudyHours vs ExamScore
plt.figure(figsize=(8, 5))
sns.scatterplot(x="StudyHours", y="ExamScore", data=df, alpha=0.6)
plt.title("StudyHours vs ExamScore", fontweight="bold")
plt.xlabel("StudyHours")
plt.ylabel("ExamScore")
plt.tight_layout()
plt.show()

# =========================
# 5. BOXPLOT THEO FINALGRADE
# =========================
features_box = ["StudyHours", "Attendance", "AssignmentCompletion", "ExamScore"]

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features_box, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="FinalGrade", y=feature, data=df)
    plt.title(f"{feature} theo FinalGrade")

plt.suptitle("So sánh các đặc trưng theo FinalGrade", fontweight="bold")
plt.tight_layout()
plt.show()

# =========================
# 6. HEATMAP – MA TRẬN TƯƠNG QUAN
# =========================
numeric_cols = [
    "StudyHours",
    "Attendance",
    "AssignmentCompletion",
    "ExamScore",
    "Motivation",
    "StressLevel",
    "FinalGrade"
]

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Heatmap ma trận tương quan", fontweight="bold")
plt.tight_layout()
plt.show()

print("✅ Hoàn thành vẽ toàn bộ biểu đồ")


















