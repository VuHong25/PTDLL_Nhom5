# ==================================================
# HUẤN LUYỆN LINEAR REGRESSION TRÊN DỮ LIỆU Z-SCORE
# ==================================================
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import os
import pickle

spark = SparkSession.builder.appName("FinalGrade_LinearRegression").getOrCreate()

# 1️⃣ Đọc dữ liệu Z-score
df = spark.read.csv("output/student_clean_zscore.csv", header=True, inferSchema=True)
print("Số dòng dữ liệu:", df.count())

# 2️⃣ Xác định label và feature
label_col = "FinalGrade"
feature_cols = [
    "StudyHours_z",
    "Attendance_z",
    "ExamScore_z",
    "AssignmentCompletion_z",
    "Discussions_z",
    "Motivation_z",
    "StressLevel_z"
]

# 3️⃣ Pipeline: chỉ assembler vì dữ liệu đã Z-score
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol=label_col, regParam=0.01, elasticNetParam=0.0)
pipeline = Pipeline(stages=[assembler, lr])

# 4️⃣ Evaluator
rmse_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
r2_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

# 5️⃣ Huấn luyện & đánh giá 10 lần
results = []
base_model_path = "finalgrade_lr_models"
os.makedirs(base_model_path, exist_ok=True)

for run in range(1, 11):
    print(f"\n===== LẦN CHẠY {run} =====")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=run*42)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    rmse = rmse_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)
    print(f"RMSE={rmse}, R2={r2}")
    results.append((run, rmse, r2))

    # Lưu model lần chạy
    lr_model = model.stages[-1]
    run_path = os.path.join(base_model_path, f"run_{run}")
    os.makedirs(run_path, exist_ok=True)
    model_data = {
        "coefficients": lr_model.coefficients.toArray(),
        "intercept": lr_model.intercept,
        "features": feature_cols
    }
    with open(os.path.join(run_path, "linear_model.pkl"), "wb") as f:
        pickle.dump(model_data, f)

# 6️⃣ Lưu kết quả đánh giá
df_results = pd.DataFrame(results, columns=["Run","RMSE","R2"])
df_results.to_csv(os.path.join(base_model_path,"evaluation_results.csv"), index=False)
print("\n===== KẾT QUẢ 10 LẦN CHẠY =====")
print(df_results)
# 7️⃣ TỔNG HỢP HỆ SỐ HỒI QUY (MEAN + STD)
# ==================================================
import numpy as np

coef_list = []
intercepts = []

for run in range(1, 11):
    with open(f"{base_model_path}/run_{run}/linear_model.pkl", "rb") as f:
        data = pickle.load(f)
        coef_list.append(data["coefficients"])
        intercepts.append(data["intercept"])

coef_array = np.array(coef_list)

coef_mean = coef_array.mean(axis=0)
coef_std = coef_array.std(axis=0)

df_coef = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient_Mean": coef_mean,
    "Coefficient_Std": coef_std
})

print("\n===== BẢNG HỆ SỐ HỒI QUY (TRUNG BÌNH 10 LẦN CHẠY) =====")
print(df_coef)

df_coef.to_csv(os.path.join(base_model_path, "regression_coefficients_mean_std.csv"), index=False)

print("\nIntercept mean:", np.mean(intercepts))
print("Intercept std:", np.std(intercepts))
spark.stop()
