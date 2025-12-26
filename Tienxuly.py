



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
# ==================================================
# 5.5XỬ LÝ NGOẠI LAI (OUTLIERS)
# ==================================================
print("\n===== XỬ LÝ NGOẠI LAI =====")

# Xác định các cột số cần kiểm tra outliers
numeric_cols_for_outlier = [c for c, t in df_clean.dtypes if t in ("int", "double")]

# Loại bỏ FinalGrade khỏi danh sách vì nó là biến phân loại
if "FinalGrade" in numeric_cols_for_outlier:
    numeric_cols_for_outlier.remove("FinalGrade")
if "FinalGrade_Goc" in numeric_cols_for_outlier:
    numeric_cols_for_outlier.remove("FinalGrade_Goc")

# Lưu số dòng trước khi xử lý
before_outlier = df_clean.count()

# IQR (Interquartile Range)
print("\nSử dụng phương pháp IQR để phát hiện outliers...")

for c in numeric_cols_for_outlier:
    # Tính Q1, Q3
    quantiles = df_clean.approxQuantile(c, [0.25, 0.75], 0.01)
    Q1 = quantiles[0]
    Q3 = quantiles[1]
    IQR = Q3 - Q1

    # Xác định ngưỡng
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Đếm số outliers
    outliers_count = df_clean.filter(
        (F.col(c) < lower_bound) | (F.col(c) > upper_bound)
    ).count()

    print(f"\n{c}:")
    print(f"  Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"  Ngưỡng: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Số outliers: {outliers_count} ({100 * outliers_count / before_outlier:.2f}%)")

    # Loại bỏ outliers
    df_clean = df_clean.filter(
        (F.col(c) >= lower_bound) & (F.col(c) <= upper_bound)
    )

after_outlier = df_clean.count()
removed = before_outlier - after_outlier

print(f"\n{'=' * 60}")
print(f"Tổng số dòng trước khi xử lý outliers: {before_outlier}")
print(f"Tổng số dòng sau khi xử lý outliers: {after_outlier}")
print(f"Số dòng bị loại bỏ: {removed} ({100 * removed / before_outlier:.2f}%)")
print(f"{'=' * 60}")

#=====
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
# ==================================================
# 6.5️⃣ PHÂN TÍCH TƯƠNG QUAN VỚI FINALGRADE
# ==================================================
print("\nTính hệ số tương quan giữa các thuộc tính và FinalGrade...")

correlations = []  # Danh sách lưu (tên thuộc tính, hệ số tương quan)

for c in numeric_cols:
    # Bỏ qua chính FinalGrade và cột gốc trước khi reverse encoding
    if c != "FinalGrade" and c != "FinalGrade_Goc":
        try:
            # Kiểm tra độ lệch chuẩn trước khi tính correlation
            std_val = df_clean.select(F.stddev(c)).first()[0]

            if std_val is None or std_val == 0:
                print(f"Bỏ qua cột '{c}': Không có sự biến thiên (std = 0)")
                correlations.append((c, 0.0))  # Gán correlation = 0
                continue

            # Tính hệ số tương quan Pearson giữa cột c và FinalGrade
            corr = df_clean.stat.corr(c, "FinalGrade")
            correlations.append((c, corr))

        except Exception as e:
            print(f"Lỗi khi tính correlation cho cột '{c}': {str(e)}")
            correlations.append((c, None))

# Lọc bỏ các giá trị None
correlations = [(c, corr) for c, corr in correlations if corr is not None]

# Sắp xếp theo độ mạnh của tương quan (giá trị tuyệt đối giảm dần)
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# Chuyển sang DataFrame Pandas để hiển thị
corr_df = pd.DataFrame(correlations, columns=["Thuộc tính", "Hệ số tương quan"])

print("\n" + "=" * 60)
print("TƯƠNG QUAN GIỮA CÁC THUỘC TÍNH VÀ FINALGRADE")
print("=" * 60)
print(corr_df.to_string(index=False))

# Ngưỡng xác định thuộc tính quan trọng
threshold = 0.1

# Lọc các thuộc tính có tương quan đáng kể
important_features = [f for f, c in correlations if abs(c) > threshold]

print(f"\nCó {len(important_features)} thuộc tính có |hệ số tương quan| > {threshold}")
if important_features:
    print(f"Danh sách: {', '.join(important_features)}")

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
