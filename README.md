# Iris: โมเดลพื้นฐานสำหรับฝึกเขียน Ensemble

ติดตั้งไลบรารีใน `.venv` และเตรียมโมเดลที่ยังไม่ผ่านการฝึกไว้แล้ว
มีตัวอย่าง Ensemble หลายชนิดที่เขียนขั้นตอนสุ่มและโหวตเองใน `bagging.py`
มี Boosting แบบ SAMME ที่เขียนการปรับน้ำหนักและโหวตเองใน `boosting.py`

## เริ่มใช้งาน

รันจากโฟลเดอร์โปรเจกต์:

```bash
source .venv/bin/activate
python iris_baseline.py
```

หากย้ายไปเครื่องใหม่ ให้สร้าง environment และติดตั้งก่อน:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

เลือก Python interpreter ของ editor เป็น `.venv/bin/python`

## เปรียบเทียบ SVM เดี่ยว กับ Bagging และ Boosting แบบเดิม

```bash
.venv/bin/python compare_models.py
```

`compare_models.py` แบ่งข้อมูลเพียงครั้งเดียวเป็น train 120 แถว / test 30 แถว
แล้วใช้ชุดเดียวกันในการเปรียบเทียบ:

| วิธี | โมเดลที่ใช้ |
| --- | --- |
| Single SVM | SVM หนึ่งตัว ฝึกจาก train ทั้งชุด |
| Bagging (4 types) | SVM, KNN, Naive Bayes, Logistic Regression อย่างละหนึ่งตัว พร้อม Bootstrap |
| Boosting (4 types) | โมเดล 4 ชนิดเดิม ฝึกเรียงกัน 4 รอบ พร้อมปรับน้ำหนัก |

ตารางแสดงจำนวนโมเดล จำนวนที่ทายถูก Accuracy, Macro F1 เวลาในการฝึก และเวลา test
Macro F1 คือค่าเฉลี่ย F1 ของทั้ง 3 สายพันธุ์ โดยให้ความสำคัญแต่ละสายพันธุ์เท่ากัน
บันทึกค่าตั้ง ผลสรุป confusion matrix และคำทำนายรายแถวไว้ใน `results/model_comparison.json`
เวลา Bagging รวมการเปิด worker ด้วย จึงอาจช้ากว่าวิธีอื่นบน Iris ที่มีข้อมูลน้อย
นี่เป็นการเปรียบเทียบ SVM เดี่ยวกับ Ensemble ผสมบน test ชุดเดียว ไม่ใช่ผลสรุปทั่วไปว่าอัลกอริทึมใดดีกว่า

เวลา test จับตั้งแต่เริ่ม `predict(X_test)` จนได้คำตอบครบ 30 แถว
รวมการปรับสเกล การทำนายทุกโมเดล และการโหวตของ Ensemble
ไม่รวมการฝึก การโหลดข้อมูล หรือการคำนวณ Accuracy และแสดงเป็นมิลลิวินาที (1 วินาที = 1,000 ms)
วัดครั้งเดียวต่อวิธี จึงอาจเปลี่ยนได้เมื่อรันใหม่ โดยเฉพาะชุดข้อมูลเล็กอย่าง Iris

สร้างกราฟ Accuracy เวลาฝึก และเวลา test จากผลที่บันทึกไว้:

```bash
.venv/bin/python plot_comparison.py
```

ได้ `results/accuracy_time_comparison.png` และ `.svg` เรียง Bagging → Boosting → SVM
และ `results/test_time_comparison.png` และ `.svg` สำหรับเวลา test โดยเฉพาะ
กราฟเวลา test เป็นกราฟแท่งเริ่มจากศูนย์ หน่วยมิลลิวินาที สำหรับข้อมูล test ทั้งชุด
กราฟเวลาฝึกใช้แกน log เพื่อแสดงค่าที่ต่างกันมาก และระบุเวลาจริงเป็นวินาทีไว้ทุกจุด
เปลี่ยน `TIME_SCALE` ใน `plot_comparison.py` เป็น `"linear"` ได้หากต้องการแกนปกติ
สคริปต์วาดกราฟอ่านผลเดิมโดยไม่ฝึกซ้ำ ถ้าต้องการผลใหม่ให้รัน `compare_models.py` ก่อน

## Ensemble หลายชนิดแบบ Parallel พร้อม Bootstrap

```bash
.venv/bin/python bagging.py
.venv/bin/python bagging.py --models svm knn naive_bayes --n-jobs 3
```

ค่าเริ่มต้นใช้ **4 ตัว คนละชนิด**: SVM, KNN, Gaussian Naive Bayes และ Logistic Regression
แต่ละชนิดมีหนึ่งตัว และใช้ worker สูงสุด 4 processes
เปลี่ยนรายชื่อได้ที่ `MODEL_NAMES` ด้านบนของ `bagging.py`
ส่วนจำนวน worker และการสุ่มปรับที่ `N_JOBS` และ `SEED`
หรือใช้ตัวเลือก command line ซึ่งจะมีผลแทนค่าด้านบนสำหรับการรันครั้งนั้น
จำนวนโมเดลเท่ากับจำนวนชื่อที่เลือก จึงไม่ใช้ `N_ESTIMATORS` หรือ `--n-estimators` แล้ว
`--model` ยังใช้เป็นชื่อแทนของ `--models` ได้ และรายชื่อโมเดลต้องไม่ซ้ำกัน

รูปแบบนี้เป็น **heterogeneous voting ensemble ร่วมกับ bootstrap sampling**
โดย Bagging แบบมาตรฐานจะใช้โมเดลชนิดเดียวกันหลายตัว
แต่ละตัวในไฟล์นี้สุ่ม 120 แถวจากชุด train แบบคืนตัวอย่างกลับ ด้วย seed ของตัวเอง
โมเดลแต่ละตัวและ scaler ของมันฝึกจาก bootstrap ของตัวเอง จากนั้นโหวตคำทำนายของ test 30 แถว
หากคะแนนโหวตเสมอ จะเลือก label ที่มีค่าน้อยที่สุด เพื่อให้ได้ผลแน่นอน

โค้ดแบ่งเป็น 3 ฟังก์ชัน:

- `train_one_bag()` สุ่มข้อมูลและฝึกหนึ่งโมเดล
- `fit_bagging()` รับ dict ชื่อ→โมเดล และเรียก `Parallel(n_jobs=...)` ให้ฝึกแต่ละชนิดพร้อมกัน
- `predict_bagging()` รวมคำทำนายด้วย majority vote

เริ่มอ่านที่ `main()` ซึ่งเรียงขั้นตอน 1–5 แล้วค่อยดูแต่ละฟังก์ชัน
ใน `fit_bagging()` ลูปจะเตรียมงานด้วย `delayed` และเริ่มทำงานขนานจริงที่
`Parallel(n_jobs=n_jobs)(training_tasks)`
ส่วน `predict_bagging()` นับคะแนนและเลือกผู้ชนะทีละตัวอย่างด้วยลูปธรรมดา

สคริปต์แสดง accuracy ของโมเดลเดี่ยวทั้ง 4 ชนิดที่ฝึกจาก train ทั้งชุด
เทียบกับผลโหวต Ensemble ที่แต่ละตัวฝึกจาก bootstrap บน test ชุดเดียวกัน พร้อมเวลาในการฝึก Ensemble
ใช้ `--n-jobs 1` เพื่อทำทีละงาน หรือ `--n-jobs -1` เพื่อใช้ CPU ทั้งหมดที่ระบบอนุญาต
เปลี่ยน `--seed` เพื่อทดลองชุด bootstrap ใหม่ โดย train/test split ยังคงเดิม
Iris มีข้อมูลน้อย การเปิด worker อาจใช้เวลามากกว่าที่ประหยัดได้ และ Ensemble อาจไม่ได้เพิ่ม accuracy
หากเลือกค่าตั้งโมเดล ให้ใช้ validation จาก train และเก็บ test ไว้วัดผลสุดท้าย

การสุ่มและโหวตเขียนด้วย NumPy ส่วนโมเดลเดี่ยวใช้ scikit-learn และการทำงานขนานใช้ joblib

## Boosting แบบ SAMME

```bash
.venv/bin/python boosting.py
.venv/bin/python boosting.py --models svm knn naive_bayes logistic_regression --rounds 8
```

ค่าเริ่มต้นฝึกตามลำดับ **SVM → KNN → Naive Bayes → Logistic Regression** รวมสูงสุด 4 รอบ
แก้ได้ที่ `MODEL_NAMES`, `N_ROUNDS` และ `SEED` ด้านบนของ `boosting.py`
ถ้ากำหนดรอบมากกว่าจำนวนชื่อ จะวนกลับไปชื่อแรกโดยสร้างโมเดลใหม่ทุกครั้ง
Boosting ต้องฝึกทีละรอบ เพราะใช้น้ำหนักที่คำนวณจากข้อผิดพลาดของรอบก่อนหน้า

โค้ดแบ่งเป็น 3 ฟังก์ชัน:

- `train_one_round()` ฝึกหนึ่งโมเดลโดยใช้ความสำคัญของแต่ละแถว
- `fit_boosting()` คำนวณข้อผิดพลาดแบบถ่วงน้ำหนัก คำนวณ alpha และเพิ่มน้ำหนักแถวที่ทายผิด
- `predict_boosting()` โหวตโดยใช้ alpha เป็นคะแนนของแต่ละโมเดล

เริ่มต้นด้วยน้ำหนักแถวละ `1/120` และรักษาผลรวมน้ำหนักเป็น 1
SVM, Naive Bayes และ Logistic Regression ใช้ train ทั้งชุดพร้อม `sample_weight`
ก่อนส่งเข้าโมเดลจะคูณน้ำหนักด้วยจำนวนแถว เพื่อให้น้ำหนักเฉลี่ยเป็น 1 และรักษาสเกล regularization
KNN ใช้การสุ่มแบบคืนตัวอย่างกลับตามน้ำหนัก (`p=sample_weights`) ซึ่งเป็นวิธีประมาณการฝึกแบบถ่วงน้ำหนัก
ทุกโมเดลมี scaler ของตัวเองที่เรียนรู้จากข้อมูลฝึกของรอบนั้น
วัด weighted error บน train ทั้งชุดเสมอ รวมถึงแถวที่ KNN ไม่ได้สุ่มไปฝึก

SAMME รองรับหลายคลาส โดยใช้สูตร:

```text
error = ผลรวมน้ำหนักของแถวที่ทายผิด
alpha = log((1 - error) / error) + log(จำนวนคลาส - 1)
น้ำหนักแถวที่ผิด = น้ำหนักเดิม × exp(alpha)
จากนั้นหารน้ำหนักทุกแถวด้วยผลรวมน้ำหนัก เพื่อให้ผลรวมกลับเป็น 1
```

นี่เป็นตัวอย่าง SAMME ที่สลับชนิดโมเดลตามรายชื่อและใช้ weighted resampling สำหรับ KNN
รอบที่ error ไม่ต่ำกว่า `1 - 1/จำนวนคลาส` จะถูกข้ามโดยไม่เพิ่มโมเดลลงผลโหวต
หากทาย train ถูกทั้งหมด จะหยุดก่อนครบจำนวนรอบ และให้น้ำหนักโหวตมากกว่าคะแนนเดิมรวมกัน
เพื่อแทนกรณี alpha เป็นอนันต์ด้วยค่าที่คำนวณได้ ถ้าทุกโมเดลไม่ผ่านเกณฑ์จะแจ้งข้อผิดพลาด
หากคะแนนรวมเสมอ จะเลือกคลาสที่มีรหัสน้อยที่สุด

Train/test เป็นชุดเดียวกับ `bagging.py` และไม่ใช้ test ในการปรับน้ำหนักหรือเลือกโมเดล
ผลลัพธ์แสดง error, alpha, จำนวนโมเดลที่ผ่านเกณฑ์, accuracy และรายงานรายสายพันธุ์
จำนวนรอบเป็นจำนวนครั้งที่ลองฝึก จำนวนโมเดลที่ใช้จริงอาจน้อยลงเมื่อข้ามรอบหรือหยุดก่อนกำหนด
ทดสอบสูตรและกรณีขอบด้วย `.venv/bin/python -m unittest discover -s tests -v`

## โมเดลที่เตรียมไว้

| งาน | เรียกใช้ | โมเดล |
| --- | --- | --- |
| ทำนาย Species | `make_classifiers()` | LogisticRegression, KNeighborsClassifier, SVC, GaussianNB |
| ทำนายตัวเลข | `make_regressors()` | LinearRegression, KNeighborsRegressor, SVR |
| จัดกลุ่ม | `make_kmeans()` | KMeans จำนวน 3 กลุ่ม |
| Genetic Algorithm | `GA` | PyGAD ต้องกำหนด fitness function เอง |

Regression ต้องใช้คำตอบเป็นตัวเลข เช่น ทำนาย `PetalLengthCm` จากอีก 3 ค่าการวัด
โดยเอาคอลัมน์คำตอบออกจาก X ด้วย ส่วน Logistic Regression ใช้ทำนาย Species ได้
K-means ให้หมายเลขกลุ่มที่ไม่ได้ตรงกับรหัส Species โดยอัตโนมัติ
GA เป็นตัวช่วยค้นหาคำตอบหรือค่าตั้ง ไม่ใช่ตัวจำแนกสายพันธุ์โดยตรง

## ตัวอย่างเรียกโมเดลเดี่ยว

```python
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from base_models import make_classifiers, make_regressors, make_kmeans, GA
from iris_baseline import load_iris_split

X_train, X_test, y_train, y_test, encoder = load_iris_split()
models = make_classifiers()

learner = Pipeline([
    ("scale", StandardScaler()),
    ("model", clone(models["svm"])),
])
learner.fit(X_train, y_train)
predictions = learner.predict(X_test)
print(encoder.inverse_transform(predictions))
```

ข้อมูลแบ่ง train/test เป็น 120/30 แถวโดยรักษาสัดส่วนสายพันธุ์ ใช้ 4 ค่าการวัดและตัด Id ออก
`load_iris_split()` คืน X ที่ยังไม่ได้ปรับสเกล และ y เป็น 0, 1, 2
Pipeline เรียนรู้สเกลจากข้อมูลที่ส่งเข้า `fit()` เท่านั้น
เมื่อสร้าง learner หลายตัว ให้ใช้ `clone()` เพื่อให้แต่ละตัวมีสถานะการฝึกแยกกัน
ถ้าใช้ validation หรือ out-of-bag ให้ fit scaler ภายในชุดฝึกของ learner แต่ละตัวด้วย

## การส่งน้ำหนักตอนฝึก Boosting

| Classifier | รับ `sample_weight` ใน `fit()` |
| --- | --- |
| LogisticRegression | ได้ |
| SVC | ได้ |
| GaussianNB | ได้ |
| KNeighborsClassifier | ไม่ได้ |

โมเดลเดี่ยวที่รองรับใช้ `model.fit(X, y, sample_weight=weights)`
สำหรับ Pipeline ที่ตั้งชื่อขั้นตอนเหมือนตัวอย่าง ใช้
`learner.fit(X, y, model__sample_weight=weights)`
โดย weights ต้องมีหนึ่งค่าต่อตัวอย่างในชุดฝึกที่ส่งเข้าไป
KNN ต้องใช้แนวทางอื่น เช่น สุ่มตัวอย่างตามน้ำหนัก หากจะทดลอง Boosting กับมัน
ค่า `weights="distance"` ของ KNN เป็นน้ำหนักตามระยะทาง ไม่ใช่น้ำหนักตัวอย่างฝึกแบบ Boosting
Iris มี 3 คลาส จึงต้องใช้ Boosting ที่รองรับหลายคลาส เช่น SAMME หากเลือกแนวทาง AdaBoost
SVC ที่เตรียมไว้รองรับ `predict()`; ถ้าต้องใช้ `predict_proba()` ให้ตั้ง `probability=True` ก่อนฝึก

## เอกสารอ้างอิง

- [scikit-learn models](https://scikit-learn.org/stable/supervised_learning.html)
- [KNN และพารามิเตอร์ fit](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [PyGAD: Genetic Algorithm](https://pygad.readthedocs.io/en/latest/)
- [Joblib Parallel](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html)
- [Voting ensemble](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
- [SAMME และสูตรน้ำหนักโหวต](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html)
