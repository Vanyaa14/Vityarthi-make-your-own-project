import subprocess
import sys
import importlib
import os
import io

required = ["pandas", "numpy", "scikit-learn"]

def silent_install(pkg):
    try:
        importlib.import_module(pkg.replace("-", ""))
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

for pkg in required:
    silent_install(pkg)

import io
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ================== EMBEDDED CSV ==================
csv_data = """Age,Race,Marital Status,T Stage ,N Stage,6th Stage,differentiate,Grade,A Stage,Tumor Size,Estrogen Status,Progesterone Status,Regional Node Examined,Reginol Node Positive,Survival Months,Status
68,White,Married,T1,N1,IIA,Poorly differentiated,3,Regional,4,Positive,Positive,24,1,60,Alive
50,White,Married,T2,N2,IIIA,Moderately differentiated,2,Regional,35,Positive,Positive,14,5,62,Alive
40,White,Married,T2,N1,IIB,Moderately differentiated,2,Regional,30,Positive,Positive,9,1,14,Dead
57,White,Single ,T3,N3,IIIC,Moderately differentiated,2,Regional,70,Positive,Positive,12,12,42,Dead
62,White,Widowed,T1,N1,IIA,Moderately differentiated,2,Regional,15,Positive,Positive,12,1,59,Alive
53,White,Married,T2,N2,IIIA,Moderately differentiated,2,Regional,32,Positive,Positive,16,9,82,Alive
69,White,Widowed,T1,N1,IIA,Moderately differentiated,2,Regional,5,Positive,Positive,8,3,82,Alive
37,White,Single ,T2,N1,IIB,Moderately differentiated,2,Regional,23,Positive,Positive,17,3,71,Alive
57,White,Single ,T3,N3,IIIC,Moderately differentiated,2,Regional,70,Positive,Positive,12,12,42,Dead
42,White,Married,T1,N3,IIIC,Moderately differentiated,2,Regional,9,Negative,Negative,15,2,39,Dead
"""

# ================== LOAD DATA FROM STRING ==================
df = pd.read_csv(io.StringIO(csv_data))
df.columns = df.columns.str.strip()
df = df.dropna()

options = {}
for col in df.select_dtypes(include="object").columns:
    if col != "Status":
        vals = sorted(df[col].dropna().unique())
        options[col] = vals

numeric_ranges = {}
for col in df.select_dtypes(include=[np.number]).columns:
    numeric_ranges[col] = (float(df[col].min()), float(df[col].max()))

y = df["Status"]
x = df.drop("Status", axis=1)

encoders = {}
x_encoded = x.copy()
for col in x.select_dtypes(include="object").columns:
    le = LabelEncoder()
    x_encoded[col] = le.fit_transform(x[col].astype(str))
    encoders[col] = le

le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y_encoded, test_size=0.2, random_state=42
)
if len(np.unique(y_train)) < 2:
    raise ValueError("Need both Alive and Dead examples in embedded CSV.")

model = LogisticRegression(max_iter=500)
model.fit(x_train, y_train)

feature_description_text = """
Feature meanings (simplified):
- Age: Age of the patient in years.
- Race: Ethnic background (e.g., White, Black, Other).
- Marital Status: Married, Single, Divorced, Widowed, etc. Used as a social factor.
- T Stage: Tumor size/extent stage (T1–T4).
- N Stage: Lymph node involvement (N0–N3).
- 6th Stage: AJCC 6th edition combined cancer stage (e.g., IIA, IIIB, IIIC).
- differentiate: Tumor cell differentiation (Well, Moderately, Poorly).
- Grade: Numeric tumor grade 1–3 or similar scale.
- A Stage: Spread extent category.
- Tumor Size: Size of the primary tumor (mm).
- Estrogen Status: Estrogen receptor positive/negative.
- Progesterone Status: Progesterone receptor positive/negative.
- Regional Node Examined: Number of lymph nodes examined.
- Reginol Node Positive: Number of nodes with cancer.
- Survival Months: Observed survival time in months.
- Status: Alive or Dead (target — used to learn likely recovery).
"""

# ================== GUI ==================
root = tk.Tk()
root.title("Cancer Survival & Condition Predictor")
root.geometry("900x900")
root.minsize(700, 700)
root.configure(bg="white")

root.rowconfigure(0, weight=0)
root.rowconfigure(1, weight=0)
root.rowconfigure(2, weight=0)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=0)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

title = tk.Label(
    root,
    text="Cancer Survival & Condition Predictor",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="black"
)
title.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

subtitle = tk.Label(
    root,
    text="Enter your clinical details. The system will estimate if patient is likely to recover (Alive) or not.",
    font=("Arial", 11),
    bg="white",
    fg="black",
    wraplength=800,
    justify="left"
)
subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 5), padx=10, sticky="ew")

desc_frame = tk.LabelFrame(root, text="Feature Description", bg="white",
                           fg="black", font=("Arial", 11, "bold"))
desc_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

desc_text = tk.Text(
    desc_frame,
    height=10,
    wrap="word",
    bg="#f7f7f7",
    fg="black",
    font=("Arial", 10)
)
desc_text.insert("1.0", feature_description_text)
desc_text.configure(state="disabled")
desc_text.pack(fill="both", expand=True, padx=5, pady=5)

form_outer = tk.Frame(root, bg="white")
form_outer.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
root.rowconfigure(3, weight=1)

canvas = tk.Canvas(form_outer, bg="white", highlightthickness=0)
scrollbar = tk.Scrollbar(form_outer, orient="vertical", command=canvas.yview)
form_frame = tk.Frame(canvas, bg="white")

form_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=form_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

entries = {}
row_pos = 0

for col in x.columns:
    lab = tk.Label(form_frame, text=f"{col}:", font=("Arial", 11),
                   bg="white", fg="black")
    lab.grid(row=row_pos, column=0, sticky="w", padx=10, pady=4)

    if col in options:
        var = tk.StringVar()
        var.set(options[col][0])
        opt = tk.OptionMenu(form_frame, var, *options[col])
        opt.config(font=("Arial", 11), fg="black", bg="white", width=20)
        opt.grid(row=row_pos, column=1, padx=5, pady=4, sticky="ew")
        entries[col] = var
    elif col in numeric_ranges:
        minv, maxv = numeric_ranges[col]
        frame = tk.Frame(form_frame, bg="white")
        ent = tk.Entry(frame, width=15, font=("Arial", 11),
                       bg="white", fg="black", insertbackground="black")
        ent.pack(side="left")
        hint = tk.Label(frame, text=f"({minv:.1f} – {maxv:.1f})",
                        font=("Arial", 9), bg="white", fg="black")
        hint.pack(side="left", padx=4)
        frame.grid(row=row_pos, column=1, padx=5, pady=4, sticky="w")
        entries[col] = ent
    else:
        ent = tk.Entry(form_frame, width=20, font=("Arial", 11),
                       bg="white", fg="black", insertbackground="black")
        ent.grid(row=row_pos, column=1, padx=5, pady=4, sticky="ew")
        entries[col] = ent

    row_pos += 1

for i in range(2):
    form_frame.columnconfigure(i, weight=1)

def predict():
    try:
        sample = {}
        for col in x.columns:
            widget = entries[col]
            if col in options:
                val = widget.get()
            else:
                val = widget.get()

            if val == "":
                messagebox.showerror("Input Error", f"Missing value for {col}.")
                return

            if col in numeric_ranges:
                try:
                    v = float(val)
                except ValueError:
                    messagebox.showerror("Input Error", f"{col}: Please enter a number.")
                    return
                minv, maxv = numeric_ranges[col]
                if not (minv <= v <= maxv):
                    messagebox.showerror(
                        "Range Error",
                        f"{col}: Value out of range. Please use between {minv:.1f} and {maxv:.1f}."
                    )
                    return
                sample[col] = [v]
            else:
                sample[col] = [val]

        sample_df = pd.DataFrame(sample)

        for col in sample_df.select_dtypes(include="object").columns:
            if col in encoders:
                le = encoders[col]
                if sample_df[col].iloc[0] not in le.classes_:
                    messagebox.showerror(
                        "Input Error",
                        f"{col}: value not seen in training data. Please choose a listed option."
                    )
                    return
                sample_df[col] = le.transform(sample_df[col].astype(str))

        pred = model.predict(sample_df)[0]
        pred_text = le_y.inverse_transform([pred])[0]

        if pred_text == "Alive":
            msg = (
                "Prediction:\n\n"
                "- The patient is likely to RECOVER / SURVIVE.\n"
                "- In this dataset, all patients have cancer; this predicts survival, not presence of cancer."
            )
        else:
            msg = (
                "Prediction:\n\n"
                "- The patient is at HIGH RISK (not likely to recover / survive).\n"
                "- This suggests advanced or aggressive cancer based on the entered features."
            )
        messagebox.showinfo("Prediction Result", msg)

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

button_frame = tk.Frame(root, bg="white")
button_frame.grid(row=4, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
root.rowconfigure(4, weight=0)

predict_btn = tk.Button(
    button_frame,
    text="Predict Recovery & Cancer Status",
    font=("Arial", 14, "bold"),
    bg="#e0e0ff",
    fg="black",
    activebackground="#c0c0ff",
    activeforeground="black",
    command=predict
)
predict_btn.pack(fill="x", expand=True)

root.mainloop()
