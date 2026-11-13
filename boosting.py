"""Boosting แบบ SAMME: ฝึกทีละรอบ → เพิ่มน้ำหนักแถวที่ผิด → โหวตแบบมีน้ำหนัก"""

import argparse
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from base_models import make_classifiers
from iris_baseline import load_iris_split


# รอบ 1 ใช้ SVM, รอบ 2 ใช้ KNN, รอบ 3 ใช้ Naive Bayes, รอบ 4 ใช้ Logistic Regression
MODEL_NAMES = ["svm", "knn", "naive_bayes", "logistic_regression"]
N_ROUNDS = 4  # ถ้าเพิ่มเป็น 8 จะวนรายชื่อโมเดลสองรอบ
SEED = 42


def train_one_round(base_model, X_train, y_train, sample_weights, seed):
    """ฝึกโมเดลใหม่โดยให้ความสำคัญกับแถวที่มีน้ำหนักสูง"""
    new_model = clone(base_model)
    if "random_state" in new_model.get_params():
        new_model.set_params(random_state=seed)

    learner = Pipeline([
        ("scale", StandardScaler()),
        ("model", new_model),
    ])

    if isinstance(new_model, KNeighborsClassifier):
        # KNN ไม่รับ sample_weight จึงสุ่มตามน้ำหนักแทน
        # แถวที่น้ำหนักสูง มีโอกาสถูกเลือกมากกว่า
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(
            len(X_train),
            size=len(X_train),
            replace=True,
            p=sample_weights,
        )
        X_bootstrap = X_train[sample_indices]
        y_bootstrap = y_train[sample_indices]
        learner.fit(X_bootstrap, y_bootstrap)
    else:
        # SVM, Naive Bayes และ Logistic Regression รับน้ำหนักได้โดยตรง
        # ทำให้น้ำหนักเฉลี่ยเป็น 1 เพื่อคงสเกลการฝึกของโมเดลที่มี regularization
        fit_weights = sample_weights * len(X_train)
        learner.fit(X_train, y_train, model__sample_weight=fit_weights)

    return learner


def fit_boosting(base_models, X_train, y_train, n_rounds=N_ROUNDS, seed=SEED):
    """คืนโมเดลที่ผ่านเกณฑ์ น้ำหนักโหวต และรายชื่อคลาส"""
    if not base_models:
        raise ValueError("At least one base model is required")
    if n_rounds < 1 or seed < 0:
        raise ValueError("Require n_rounds >= 1 and seed >= 0")

    classes = np.unique(y_train)
    number_of_classes = len(classes)
    if number_of_classes < 2:
        raise ValueError("Boosting requires at least two classes")

    # เริ่มต้น: ทุกแถวสำคัญเท่ากัน น้ำหนักรวมเท่ากับ 1
    number_of_samples = len(y_train)
    sample_weights = np.full(number_of_samples, 1.0 / number_of_samples)
    model_names = list(base_models)
    models = []
    model_weights = []

    # ฝึกทีละรอบ เพราะรอบถัดไปต้องใช้น้ำหนักที่อัปเดตจากรอบนี้
    for round_number in range(n_rounds):
        # % ทำให้วนกลับไปชื่อแรกเมื่อใช้ครบทุกชนิด
        model_name = model_names[round_number % len(model_names)]
        base_model = base_models[model_name]

        # กันน้ำหนักเล็กจนกลายเป็นศูนย์เมื่อทดลองหลายรอบ
        sample_weights = np.maximum(sample_weights, np.finfo(float).eps)
        sample_weights = sample_weights / sample_weights.sum()

        # 1. ฝึกตามน้ำหนักปัจจุบัน
        learner = train_one_round(
            base_model, X_train, y_train,
            sample_weights, seed + round_number,
        )

        # 2. ตรวจบน train ทั้งชุด รวมถึงแถวที่ KNN อาจไม่ได้สุ่มไปฝึก
        train_predictions = learner.predict(X_train)
        incorrect = train_predictions != y_train
        weighted_error = sample_weights[incorrect].sum()

        # Iris มี 3 คลาส: โมเดลต้องมี weighted error ต่ำกว่า 2/3
        error_limit = 1.0 - 1.0 / number_of_classes
        if weighted_error >= error_limit - 1e-12:
            print(f"Round {round_number + 1}: {model_name} | error={weighted_error:.4f} | skipped")
            continue

        # 3. คำนวณ alpha: น้ำหนักคะแนนโหวตของโมเดลนี้
        perfect_fit = not incorrect.any()
        if perfect_fit:
            # เมื่อทาย train ถูกทุกแถว ให้โมเดลนี้ชนะคะแนนเดิมทั้งหมด แล้วหยุด
            # ใช้ค่าจำกัดแทน alpha ที่จะเป็นอนันต์เมื่อ error = 0
            alpha = sum(model_weights) + 1.0
        else:
            safe_error = max(weighted_error, 1e-12)
            alpha = (
                np.log((1.0 - safe_error) / safe_error)
                + np.log(number_of_classes - 1)
            )

        models.append(learner)
        model_weights.append(alpha)
        print(f"Round {round_number + 1}: {model_name} | error={weighted_error:.4f} | alpha={alpha:.4f}")

        if perfect_fit:
            print("Perfect training predictions: stop early.")
            break

        # 4. เพิ่มน้ำหนักแถวที่ทายผิด แล้วปรับผลรวมน้ำหนักกลับเป็น 1
        sample_weights[incorrect] *= np.exp(alpha)
        sample_weights = sample_weights / sample_weights.sum()

    if not models:
        raise ValueError("No model passed the SAMME error threshold")
    return models, np.array(model_weights), classes


def predict_boosting(models, model_weights, classes, X):
    """รวมคะแนนโหวต โดยแต่ละโมเดลมีคะแนนเท่ากับ alpha ของตัวเอง"""
    if not models or len(models) != len(model_weights):
        raise ValueError("Each trained model must have one voting weight")

    # หนึ่งแถวต่อตัวอย่าง หนึ่งคอลัมน์ต่อคลาส
    class_scores = np.zeros((len(X), len(classes)))

    for model, alpha in zip(models, model_weights):
        predictions = model.predict(X)
        for class_index, class_label in enumerate(classes):
            voted_for_class = predictions == class_label
            class_scores[:, class_index] += alpha * voted_for_class

    # ถ้าเสมอ เลือกคลาสที่อยู่ก่อนใน classes ซึ่ง fit_boosting เรียงไว้แล้ว
    winning_indices = np.argmax(class_scores, axis=1)
    return classes[winning_indices]


def read_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=list(make_classifiers()), default=MODEL_NAMES,
    )
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--seed", type=int, default=SEED)
    options = parser.parse_args()
    if not options.models or len(options.models) != len(set(options.models)):
        parser.error("Specify at least one model, with no duplicate names")
    if options.rounds < 1 or options.seed < 0:
        parser.error("Require rounds >= 1 and seed >= 0")
    return options


def main():
    options = read_options()

    # 1. ใช้ train/test ชุดเดียวกับ bagging.py
    X_train, X_test, y_train, y_test, encoder = load_iris_split()
    available_models = make_classifiers(random_state=options.seed)
    selected_models = {}
    for model_name in options.models:
        selected_models[model_name] = available_models[model_name]

    # 2. ฝึกทีละรอบ และแสดง weighted error กับ alpha ของแต่ละรอบ
    print(f"Model order: {' -> '.join(options.models)}")
    start_time = perf_counter()
    models, model_weights, classes = fit_boosting(
        selected_models, X_train, y_train,
        n_rounds=options.rounds, seed=options.seed,
    )
    training_seconds = perf_counter() - start_time

    # 3. ใช้ test หลังฝึกเสร็จแล้ว เพื่อวัดผลสุดท้าย
    predictions = predict_boosting(models, model_weights, classes, X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accepted models: {len(models)} | Requested rounds: {options.rounds}")
    print(f"Train: {len(y_train)} | Test: {len(y_test)} | Features: {X_train.shape[1]}")
    print(f"Boosting accuracy: {accuracy:.3f}")
    print(f"Training time: {training_seconds:.3f} seconds")
    report = classification_report(
        y_test, predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, zero_division=0,
    )
    print(report)


if __name__ == "__main__":
    main()
