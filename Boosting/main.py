"""Manual AdaBoost (SAMME) บน Iris — รันด้วย python main.py"""
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://huggingface.co/datasets/scikit-learn/iris/resolve/main/Iris.csv"
ROUNDS = 30


class ManualAdaBoost:
    def __init__(self, n_estimators=ROUNDS):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.n_estimators < 1 or len(self.classes_) < 2:
            raise ValueError("ต้องมีอย่างน้อย 1 รอบ และ 2 คลาส")
        weights = np.full(len(y), 1 / len(y))
        self.models_, self.alphas_, self.history_ = [], [], []
        # รอบ 0 คือน้ำหนักเริ่มต้น; รอบ t คือน้ำหนักหลังจบรอบ t
        self.sample_weights_ = [weights.copy()]
        self.stop_reason_ = "ครบจำนวนรอบ"

        for step in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=1, random_state=42 + step)
            tree.fit(X, y, sample_weight=weights)
            wrong = tree.predict(X) != y
            error = float(np.sum(weights * wrong))
            limit = 1 - 1 / len(self.classes_)
            if error >= limit or np.isclose(error, limit, rtol=0, atol=1e-12):
                self.stop_reason_ = "โมเดลย่อยไม่ดีกว่าการสุ่ม"
                break

            if error == 0:
                # SAMME ให้ alpha เป็นอนันต์: ใช้ค่าจำกัดที่ชนะคะแนนเดิมทั้งหมดแทน
                # จึงเก็บโมเดลเก่าครบ และให้ต้นไม้ที่ถูกทั้งหมดเป็นผู้ตัดสิน
                alpha = sum(self.alphas_) + 1.0
            else:
                alpha = np.log((1 - error) / error) + np.log(len(self.classes_) - 1)
            self.models_.append(tree)
            self.alphas_.append(alpha)

            if error > 0:
                weights *= np.exp(alpha * wrong)  # เพิ่มน้ำหนักตัวอย่างที่ผิด
                weights /= weights.sum()         # ปรับผลรวมกลับเป็น 1
            self.sample_weights_.append(weights.copy())
            # Training Error ของ ensemble ทั้งหมดถึงรอบนี้ ไม่ใช่ weighted error ของต้นไม้
            training_error = np.mean(self.predict(X) != y)
            self.history_.append([step + 1, error, alpha, training_error])
            if error == 0:
                self.stop_reason_ = "โมเดลย่อยทายชุดฝึกถูกทั้งหมด"
                break

        if not self.models_:
            raise ValueError("โมเดลย่อยรอบแรกไม่ดีกว่าการสุ่ม จึงเริ่ม Boosting ไม่ได้")
        return self

    def predict(self, X):
        if not getattr(self, "models_", []):
            raise ValueError("ต้อง fit ให้สำเร็จก่อน predict")
        votes = np.zeros((len(X), len(self.classes_)))
        for tree, alpha in zip(self.models_, self.alphas_):
            prediction = tree.predict(X)
            for column, label in enumerate(self.classes_):
                votes[:, column] += alpha * (prediction == label)
        return self.classes_[votes.argmax(axis=1)]


def main():
    # 1. โหลดข้อมูลจาก Hugging Face และเก็บไว้สำหรับรันครั้งต่อไป
    data_path = ROOT / "Iris.csv"
    if not data_path.exists():
        pd.read_csv(DATA_URL).to_csv(data_path, index=False)
    df = pd.read_csv(data_path)
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df[features].to_numpy()  # ไม่ใช้ Id เพราะเป็นเพียงหมายเลขแถว
    y = df["Species"].to_numpy()

    # 2. แบ่ง 80/20 โดยรักษาสัดส่วนแต่ละสายพันธุ์
    train, test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    # 3. ฝึก Boosting เอง ไม่ใช้ AdaBoostClassifier
    model = ManualAdaBoost(n_estimators=ROUNDS)
    start = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start
    prediction = model.predict(X_test)
    classes = model.classes_

    # 4. เปรียบเทียบกับต้นไม้เดี่ยวที่มีความลึกเท่ากัน
    single_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    single_tree.fit(X_train, y_train)
    baseline = accuracy_score(y_test, single_tree.predict(X_test))
    accuracy = accuracy_score(y_test, prediction)
    report = classification_report(y_test, prediction, labels=classes, zero_division=0)
    matrix = pd.DataFrame(confusion_matrix(y_test, prediction, labels=classes),
                          index=classes, columns=classes)

    # 5. บันทึก error และ alpha แต่ละรอบ พร้อมผลทำนาย
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    rounds = pd.DataFrame(model.history_, columns=["round", "weighted_error", "alpha", "training_error"])
    rounds.to_csv(output / "rounds.csv", index=False)
    weight_table = pd.DataFrame(model.sample_weights_, columns=df.iloc[train].Id.to_numpy())
    weight_table.index.name = "round"
    weight_table.to_csv(output / "sample_weights.csv")

    # กราฟใช้เฉพาะ training error ไม่ใช้คะแนน test เลือกจำนวนรอบ
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "__pycache__" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds["round"], rounds["training_error"], marker="o", markersize=3)
    ax.set(xlabel="Boosting round", ylabel="Training error (ensemble)",
           title="Manual AdaBoost SAMME", ylim=(0, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "training_error.png", dpi=160)
    plt.close(fig)

    results = df.iloc[test][["Id", "Species"]].copy()
    results["prediction"] = prediction
    results.to_csv(output / "predictions.csv", index=False)
    summary = (
        f"Train: {len(train)}, Test: {len(test)}\n"
        f"Manual SAMME: {len(model.models_)} trees (max_depth=1)\n"
        f"Training time (fit only): {training_seconds:.6f} seconds\n"
        f"Single tree accuracy: {baseline:.2%}\n"
        f"Boosting accuracy: {accuracy:.2%}\n\n"
        f"{report}\nConfusion matrix (rows=actual, columns=predicted):\n{matrix}\n"
    )
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(rounds.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
