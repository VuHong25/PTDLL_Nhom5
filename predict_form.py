# ==================================================
# TKINTER – DỰ BÁO KẾT QUẢ HỌC TẬP
# ==================================================
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np

# ================= LOAD MODEL =================
MODEL_PATH = r"finalgrade_lr_models/run_1/linear_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

coefficients = model["coefficients"]
intercept = model["intercept"]
FEATURES = model["features"]

# ================= THỐNG KÊ Z-SCORE =================
stats = {
    "StudyHours": {"mean":20.03,"std":6.05},
    "Attendance":{"mean":80.24,"std":11.47},
    "ExamScore":{"mean":70.31,"std":17.70},
    "AssignmentCompletion":{"mean":74.52,"std":14.66},
    "Discussions":{"mean":0.61,"std":0.49},
    "Motivation":{"mean":0.91,"std":0.71},
    "StressLevel":{"mean":1.31,"std":0.79}
}

labels = {
    "StudyHours_z":"Số giờ học mỗi tuần",
    "Attendance_z":"Tỷ lệ chuyên cần (%)",
    "ExamScore_z":"Điểm kiểm tra",
    "AssignmentCompletion_z":"Hoàn thành bài tập (%)",
    "Discussions_z":"Tham gia thảo luận (0/1)",
    "Motivation_z":"Động lực học tập (0–2)",
    "StressLevel_z":"Mức độ căng thẳng (0–2)"
}

# ================= TKINTER =================
root = tk.Tk()
root.title("Dự báo kết quả học tập")
root.geometry("420x520")

tk.Label(root, text="DỰ BÁO KẾT QUẢ HỌC TẬP", font=("Arial", 14, "bold")).pack(pady=10)

form = tk.Frame(root)
form.pack(pady=5)

entries = {}
for i, f_z in enumerate(FEATURES):
    tk.Label(form, text=labels[f_z]).grid(row=i, column=0, sticky="w", padx=5, pady=4)
    e = tk.Entry(form, width=10)
    e.grid(row=i, column=1, pady=4)
    entries[f_z] = e

result_label = tk.Label(root, font=("Arial", 14, "bold"))
result_label.pack(pady=20)

# ================= LOGIC =================
def predict():
    try:
        x = []
        for f_z in FEATURES:
            f = f_z.replace("_z","")
            val = float(entries[f_z].get())
            mean = stats[f]["mean"]
            std = stats[f]["std"]
            z = (val - mean) / std
            x.append(z)

        x = np.array(x)
        y = np.dot(x, coefficients) + intercept
        y = max(0, min(3, y))

        result_label.config(text=f"🎯 FinalGrade dự đoán ≈ {y:.2f}")
    except:
        messagebox.showerror("Lỗi", "Vui lòng nhập đầy đủ và đúng định dạng số")

# ================= BUTTON =================
tk.Button(root, text="🔮 Dự báo", width=20, command=predict).pack(pady=10)

root.mainloop()
