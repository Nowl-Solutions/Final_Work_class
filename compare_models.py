"""เปรียบเทียบ SVM เดี่ยว กับ Bagging และ Boosting ที่ใช้โมเดล 4 ชนิด"""

import json
from pathlib import Path
from time import perf_counter

from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bagging import fit_bagging, predict_bagging
from base_models import make_classifiers
from boosting import fit_boosting, predict_boosting
from iris_baseline import FEATURES, load_iris_split


MODEL_NAMES = ["svm", "knn", "naive_bayes", "logistic_regression"]
N_JOBS = 4
SEED = 42
RESULT_PATH = Path(__file__).resolve().parent / "results" / "model_comparison.json"


def evaluate(method, predictions, y_test, model_count, training_seconds, test_seconds):
    """วัดทุกวิธีด้วยเกณฑ์เดียวกัน และเก็บคำทำนายไว้ตรวจสอบได้"""
    return {
        "method": method,
        "model_count": model_count,
        "correct": int((predictions == y_test).sum()),
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "predictions": predictions.tolist(),
    }


def run_comparison(n_jobs=N_JOBS, seed=SEED):
    # 1. แบ่งข้อมูลครั้งเดียว แล้วใช้ train/test ชุดนี้กับทั้ง 3 วิธี
    X_train, X_test, y_train, y_test, encoder = load_iris_split(random_state=seed)
    available_models = make_classifiers(random_state=seed)
    selected_models = {}
    for model_name in MODEL_NAMES:
        selected_models[model_name] = available_models[model_name]
    results = []

    # 2. SVM เดี่ยว: ฝึกจาก train ทั้ง 120 แถว
    single_svm = Pipeline([
        ("scale", StandardScaler()),
        ("model", clone(available_models["svm"])),
    ])
    start_time = perf_counter()
    single_svm.fit(X_train, y_train)
    training_seconds = perf_counter() - start_time

    # จับเวลาเฉพาะการทำนาย test: รวม scaler แต่ไม่รวมการฝึกหรือคำนวณ Accuracy
    start_time = perf_counter()
    predictions = single_svm.predict(X_test)
    test_seconds = perf_counter() - start_time
    results.append(evaluate(
        "Single SVM", predictions, y_test, 1, training_seconds, test_seconds,
    ))

    # 3. Bagging แบบเดิม: 4 ชนิด สุ่มข้อมูลแยกกัน ฝึกขนาน แล้วโหวต
    start_time = perf_counter()
    bagging_models = fit_bagging(
        selected_models, X_train, y_train, n_jobs=n_jobs, seed=seed,
    )
    training_seconds = perf_counter() - start_time

    # เวลา test ของ Ensemble รวมการทำนายทุกโมเดลและการรวมคะแนนโหวต
    start_time = perf_counter()
    predictions = predict_bagging(bagging_models, X_test)
    test_seconds = perf_counter() - start_time
    results.append(evaluate(
        "Bagging (4 types)", predictions, y_test, len(bagging_models),
        training_seconds, test_seconds,
    ))

    # 4. Boosting แบบเดิม: 4 ชนิด ฝึกทีละรอบ ปรับน้ำหนัก แล้วโหวตด้วย alpha
    start_time = perf_counter()
    boosting_models, model_weights, classes = fit_boosting(
        selected_models, X_train, y_train, n_rounds=len(MODEL_NAMES), seed=seed,
    )
    training_seconds = perf_counter() - start_time

    start_time = perf_counter()
    predictions = predict_boosting(boosting_models, model_weights, classes, X_test)
    test_seconds = perf_counter() - start_time
    results.append(evaluate(
        "Boosting (4 types)", predictions, y_test, len(boosting_models),
        training_seconds, test_seconds,
    ))

    return {
        "dataset": "scikit-learn/iris",
        "seed": seed,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "features": FEATURES,
        "classes": encoder.classes_.tolist(),
        "ensemble_models": MODEL_NAMES,
        "bagging_n_jobs": n_jobs,
        "boosting_requested_rounds": len(MODEL_NAMES),
        "boosting_model_weights": model_weights.tolist(),
        "y_test": y_test.tolist(),
        "results": results,
    }


def main():
    report = run_comparison()

    # 5. แสดงตารางเปรียบเทียบ และบันทึกผลพร้อมคำทำนายรายแถว
    print(f"\nSame split: train={report['train_samples']}, test={report['test_samples']}")
    print("Ensemble models: " + ", ".join(report["ensemble_models"]))
    print(
        f"{'Method':<23} {'Models':>6} {'Correct':>8} {'Accuracy':>10} "
        f"{'Macro F1':>10} {'Train (s)':>10} {'Test (ms)':>11}"
    )
    for result in report["results"]:
        correct = f"{result['correct']}/{result['test_samples']}"
        print(
            f"{result['method']:<23} {result['model_count']:>6} {correct:>8} "
            f"{result['accuracy']:>9.2%} {result['macro_f1']:>10.4f} "
            f"{result['training_seconds']:>10.3f} {result['test_seconds'] * 1000:>11.3f}"
        )
    print("Bagging training time includes worker startup; this is one test split.")
    print("Test time: predict all test rows, including scaling and ensemble voting (one run).")

    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
