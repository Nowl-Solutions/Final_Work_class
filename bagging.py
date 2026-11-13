"""Ensemble หลายชนิดสำหรับ Iris: สุ่มข้อมูล → ฝึกขนาน → โหวตคำตอบ"""

import argparse
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from base_models import make_classifiers
from iris_baseline import load_iris_split


# เปลี่ยนค่าตรงนี้เพื่อทดลอง หรือระบุผ่าน command line
MODEL_NAMES = [
    "svm",
    "knn",
    "naive_bayes",
    "logistic_regression",
]  # ใช้ชนิดละ 1 ตัว รวม 4 ตัว
N_JOBS = 4          # จำนวน worker: 1 = ทำทีละงาน, -1 = ใช้ CPU ที่มีทั้งหมด
SEED = 42          # ใช้ค่าเดิมเพื่อให้สุ่มได้ผลเดิม


def train_one_bag(base_model, X_train, y_train, seed):
    """สุ่มข้อมูลแบบคืนตัวอย่างกลับ แล้วฝึกโมเดลใหม่หนึ่งตัว"""
    # 1. สุ่มหมายเลขแถว: จำนวนแถวเท่าเดิม แต่มีแถวซ้ำได้
    rng = np.random.default_rng(seed)
    number_of_samples = len(X_train)
    sample_indices = rng.choice(
        number_of_samples,
        size=number_of_samples,
        replace=True,
    )

    # ใช้หมายเลขแถวเดียวกัน เพื่อให้ข้อมูลและคำตอบยังจับคู่ถูกต้อง
    X_bootstrap = X_train[sample_indices]
    y_bootstrap = y_train[sample_indices]

    # 2. สร้างโมเดลใหม่ที่ยังไม่ผ่านการฝึก
    new_model = clone(base_model)
    if "random_state" in new_model.get_params():
        new_model.set_params(random_state=seed)

    # 3. ปรับสเกลและฝึก โดยใช้เฉพาะข้อมูลที่สุ่มได้ใน bag นี้
    learner = Pipeline([
        ("scale", StandardScaler()),
        ("model", new_model),
    ])
    learner.fit(X_bootstrap, y_bootstrap)
    return learner


def fit_bagging(
    base_models, X_train, y_train,
    n_jobs=N_JOBS, seed=SEED,
):
    """รับ dict ชื่อ→โมเดล แล้วฝึกชนิดละหนึ่งตัวตามลำดับใน dict"""
    if not base_models:
        raise ValueError("At least one base model is required")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    # เตรียมงานคนละชนิด: SVM, KNN, Naive Bayes, Logistic Regression
    # delayed ยังไม่ฝึกทันที และแต่ละงานใช้ seed ของตัวเอง
    training_tasks = []
    for model_number, base_model in enumerate(base_models.values()):
        model_seed = seed + model_number
        task = delayed(train_one_bag)(
            base_model, X_train, y_train, model_seed,
        )
        training_tasks.append(task)

    # PARALLEL COMPUTING: ส่งงานให้หลาย worker ทำพร้อมกัน แล้วรอผลครบ
    # loky ใช้หลาย process; จำกัด threads ในแต่ละ worker ไม่ให้แย่ง CPU กัน
    with parallel_config(backend="loky", inner_max_num_threads=1):
        models = Parallel(n_jobs=n_jobs)(training_tasks)
    return models


def predict_bagging(models, X):
    """โหวตคลาสที่พบบ่อยที่สุด; ถ้าเสมอ เลือก label ที่เรียงก่อน"""
    if not models:
        raise ValueError("At least one trained model is required")

    # ให้ทุกโมเดลทำนายข้อมูลชุดเดียวกัน
    all_predictions = []
    for model in models:
        predictions = model.predict(X)
        all_predictions.append(predictions)

    # เดิม: หนึ่งแถวต่อโมเดล → หลัง .T: หนึ่งแถวต่อตัวอย่างข้อมูล
    votes_per_sample = np.array(all_predictions).T

    final_predictions = []
    for votes in votes_per_sample:
        # เช่น [0, 0, 1] → คลาส [0, 1] ได้คะแนน [2, 1] → เลือก 0
        labels, vote_counts = np.unique(votes, return_counts=True)
        winner_index = np.argmax(vote_counts)
        winning_label = labels[winner_index]
        final_predictions.append(winning_label)

    return np.array(final_predictions)


def read_options():
    """อ่านตัวเลือกจากคำสั่งรัน ถ้าไม่ระบุจะใช้ค่าที่ตั้งไว้ด้านบน"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", "--model",
        nargs="+",
        choices=list(make_classifiers()),
        default=MODEL_NAMES,
        help="รายชื่อโมเดลที่จะฝึก ชนิดละหนึ่งตัว",
    )
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--seed", type=int, default=SEED)
    options = parser.parse_args()

    if not options.models or len(options.models) != len(set(options.models)):
        parser.error("Specify at least one model, with no duplicate model names")
    if options.n_jobs == 0 or options.seed < 0:
        parser.error("Require n-jobs != 0 and seed >= 0")
    return options


def main():
    options = read_options()

    # 1. โหลดข้อมูล: train 120 แถว / test 30 แถว โดยแบ่งชุดเดิมทุกครั้ง
    X_train, X_test, y_train, y_test, encoder = load_iris_split()
    available_models = make_classifiers(random_state=options.seed)
    selected_models = {}
    for model_name in options.models:
        selected_models[model_name] = available_models[model_name]

    # 2. ฝึกโมเดลเดี่ยวแต่ละชนิดบน train ทั้งชุด ไว้เปรียบเทียบกับ Ensemble
    single_accuracies = {}
    for model_name, base_model in selected_models.items():
        single_model = Pipeline([
            ("scale", StandardScaler()),
            ("model", clone(base_model)),
        ])
        single_model.fit(X_train, y_train)
        single_predictions = single_model.predict(X_test)
        single_accuracies[model_name] = accuracy_score(y_test, single_predictions)

    # 3. ฝึกโมเดลต่างชนิดพร้อมกัน แต่ละตัวใช้ bootstrap ของตัวเอง
    start_time = perf_counter()
    models = fit_bagging(
        selected_models, X_train, y_train,
        n_jobs=options.n_jobs,
        seed=options.seed,
    )
    training_seconds = perf_counter() - start_time

    # 4. ให้ทุกโมเดลทำนาย test แล้วโหวตคำตอบ
    ensemble_predictions = predict_bagging(models, X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

    # 5. แสดงผล โดยโมเดลเดี่ยวและ Ensemble ใช้ test ชุดเดียวกัน
    print(f"Models: {', '.join(options.models)}")
    print(f"Total models: {len(models)} | n_jobs: {options.n_jobs}")
    print(f"Train: {len(y_train)} | Test: {len(y_test)} | Features: {X_train.shape[1]}")
    print("Single models (trained on the full training set):")
    for model_name, accuracy in single_accuracies.items():
        print(f"  {model_name}: {accuracy:.3f}")
    print(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
    print(f"Ensemble training time: {training_seconds:.3f} seconds (includes worker startup)")

    report = classification_report(
        y_test,
        ensemble_predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    )
    print(report)


if __name__ == "__main__":
    main()
