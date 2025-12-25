
from pyspark.sql import SparkSession, functions as F
import pandas as pd
import os

# ==================================================
# 1️⃣ KHỞI TẠO SPARK
# ==================================================
spark = SparkSession.builder \
    .appName("StudentPerformance_Preprocessing") \
    .getOrCreate()

# ==================================================
# 2️⃣ ĐỌC DỮ LIỆU
# ==================================================
df = spark.read.csv("student_performance.csv", header=True, inferSchema=True)

# ==================================================
# 3️⃣ KIỂM TRA GIÁ TRỊ THIẾU
# ==================================================
missing_data = []
for col_name, dtype in df.dtypes:
    missing = df.filter(F.col(col_name).isNull()).count()
    missing_data.append([col_name, dtype, missing])
missing_df = pd.DataFrame(missing_data, columns=["Tên cột", "Kiểu dữ liệu", "Số giá trị thiếu"])
print("\n===== BẢNG KIỂM TRA GIÁ TRỊ THIẾU =====")
print(missing_df)

# ==================================================
# 4KIỂM TRA TRÙNG LẶP
# ==================================================
total_rows = df.count()
dup_rows = total_rows - df.dropDuplicates().count()
print("\n===== DÒNG TRÙNG LẶP =====")
print(f"Tổng số dòng: {total_rows}, Số dòng trùng: {dup_rows}")

# 5️LÀM SẠCH DỮ LIỆU
# ==================================================
df_clean = df.dropDuplicates().dropna()
print("\n===== DỮ LIỆU SAU LÀM SẠCH =====")
print(f"Số dòng còn lại: {df_clean.count()}")
#=----
# ==================================================
# 5XỬ LÝ MÃ HÓA NGƯỢC CỦA FINALGRADE
# ==================================================
print("\nKiểm tra cách mã hóa của FinalGrade...")

correlation = df_clean.stat.corr("ExamScore", "FinalGrade")
print(f"Hệ số tương quan giữa ExamScore và FinalGrade: {correlation:.3f}")

if correlation < -0.5:
    print("\nCẢNH BÁO: Phát hiện mối tương quan âm mạnh!")
    print("   → FinalGrade có khả năng đang bị mã hóa ngược")
    print("   → Thực hiện đảo mã: 0→3, 1→2, 2→1, 3→0")

    df_clean = df_clean.withColumn(
        "FinalGrade_Goc",
        F.col("FinalGrade")
    ).withColumn(
        "FinalGrade",
        F.lit(3) - F.col("FinalGrade")
    )

    new_corr = df_clean.stat.corr("ExamScore", "FinalGrade")
    print(f"   → Hệ số tương quan sau khi đảo mã: {new_corr:.3f}")
    print("Đã đảo mã FinalGrade thành công")
else:
    print("Cách mã hóa FinalGrade là đúng")
#====
# ==================================================
# 5.6PHÂN TÍCH PHÂN PHỐI FINALGRADE
# ==================================================
print("\nPhân tích phân phối của FinalGrade...")

grade_dist = df_clean.groupBy("FinalGrade") \
    .count() \
    .orderBy("FinalGrade") \
    .toPandas()

print("\n" + "="*60)
print("PHÂN PHỐI FINALGRADE")
print("="*60)
print(grade_dist.to_string(index=False))

max_count = grade_dist['count'].max()
min_count = grade_dist['count'].min()
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
print(f"\nTỷ lệ mất cân bằng (Imbalance ratio): {imbalance_ratio:.2f}")

if imbalance_ratio > 3:
    print("⚠Dữ liệu bị mất cân bằng! Nên cân nhắc các kỹ thuật resampling hoặc gán trọng số lớp")
else:
    print("Dữ liệu tương đối cân bằng")

# ==================================================
# 6️⃣ XÁC ĐỊNH CÁC CỘT SỐ
# ==================================================
numeric_cols = [c for c, t in df_clean.dtypes if t in ("int", "double")]

# ==================================================
# 7️CHUẨN HÓA Z-SCORE
# ==================================================
df_zscore = df_clean
zscore_cols = []  # lưu tên các cột Z-score

for c in numeric_cols:
    stats_col = df_clean.select(
        F.mean(c).alias("mean"),
        F.stddev(c).alias("std")
    ).first()
    mean_val = stats_col["mean"]
    std_val = stats_col["std"] if stats_col["std"] != 0 else 1.0  # tránh chia 0

    z_col = f"{c}_z"
    zscore_cols.append(z_col)
    df_zscore = df_zscore.withColumn(z_col, ((F.col(c) - mean_val) / std_val).cast("double"))

print("\n===== MỘT SỐ DÒNG Z-SCORE =====")
df_zscore.select(zscore_cols).show(5)
#===
# 6.5️⃣ PHÂN TÍCH TƯƠNG QUAN VỚI FINALGRADE
# ==================================================
print("Tính hệ số tương quan giữa các thuộc tính và FinalGrade...")

correlations = []  # Danh sách lưu (tên thuộc tính, hệ số tương quan)

for c in numeric_cols:
    # Bỏ qua chính FinalGrade và cột gốc trước khi reverse encoding
    if c != "FinalGrade" and c != "FinalGrade_Original":
        # Tính hệ số tương quan Pearson giữa cột c và FinalGrade
        corr = df_clean.stat.corr(c, "FinalGrade")
        correlations.append((c, corr))

# Sắp xếp theo độ mạnh của tương quan (giá trị tuyệt đối giảm dần)
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# Chuyển sang DataFrame Pandas để hiển thị
corr_df = pd.DataFrame(correlations, columns=["Thuộc tính", "Hệ số tương quan"])

print("\n" + "=" * 60)
print("TƯƠNG QUAN GIỮA CÁC THUỘC TÍNH VÀ FINALGRADE (TOP 10)")
print("=" * 60)
print(corr_df.head(10).to_string(index=False))

# Ngưỡng xác định thuộc tính quan trọng
threshold = 0.1

# Lọc các thuộc tính có tương quan đáng kể
important_features = [f for f, c in correlations if abs(c) > threshold]

print(f"\nCó {len(important_features)} thuộc tính có |hệ số tương quan| > {threshold}")

# ==================================================
# 8️⃣ TẠO BẢNG PHÂN TÍCH MÔ TẢ
# ==================================================
stats = {}
stats[""] = ["Min", "Max", "Mean", "Var", "STD", "Q1", "Q2", "Q3"]

for c in numeric_cols:
    s = df_clean.select(
        F.min(c).alias("Min"),
        F.max(c).alias("Max"),
        F.mean(c).alias("Mean"),
        F.variance(c).alias("Var"),
        F.stddev(c).alias("STD"),
        F.expr(f"percentile_approx({c}, array(0.25,0.5,0.75))").alias("Q")
    ).first()

    stats[c] = [
        round(s["Min"], 2),
        round(s["Max"], 2),
        round(s["Mean"], 2),
        round(s["Var"], 2),
        round(s["STD"], 2),
        round(s["Q"][0], 2),
        round(s["Q"][1], 2),
        round(s["Q"][2], 2)
    ]

desc_df = pd.DataFrame(stats).set_index("").transpose()
print("\n===== BẢNG PHÂN TÍCH MÔ TẢ =====")
print(desc_df)

# ==================================================
# 9️⃣ XUẤT CSV
# ==================================================
os.makedirs("output", exist_ok=True)

# 9.1 Xuất dữ liệu gốc đã làm sạch
df_clean.toPandas().to_csv(
    "output/student_clean.csv",
    index=False,
    encoding="utf-8-sig"
)

# 9.2 Xuất dữ liệu chuẩn hóa Z-score
df_zscore.toPandas().to_csv(
    "output/student_clean_zscore.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n Đã xuất file: output/student_clean.csv")
print(" Đã xuất file chuẩn hóa Z-score: output/student_clean_zscore.csv")



# ==================================================
# 10️⃣ KẾT THÚC
# ==================================================
spark.stop()
