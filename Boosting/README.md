# Manual Boosting — Iris

โค้ดทั้งหมดอยู่ใน `main.py` ใช้ AdaBoost แบบ **SAMME** เพราะ Iris มี 3 คลาส
มี `test_main.py` อีกไฟล์สำหรับ Unit Tests โดยไม่แยกโค้ดใช้งานออกเป็นหลายไฟล์
ใช้ `DecisionTreeClassifier(max_depth=1)` เป็นโมเดลย่อย ส่วนขั้นตอน Boosting เขียนเอง
ไม่ได้ใช้ `AdaBoostClassifier` หรือโมเดล ensemble สำเร็จรูป

## วิธีรัน

```powershell
pip install -r requirements.txt
python main.py
python -m unittest -v
```

ทดสอบด้วย Python 3.13 ครั้งแรกต้องใช้อินเทอร์เน็ต แล้วเก็บข้อมูลใน `Iris.csv`
โหลดจาก [scikit-learn/iris บน Hugging Face](https://huggingface.co/datasets/scikit-learn/iris) (CC0-1.0)
ใช้ความยาว/ความกว้างของกลีบเลี้ยงและกลีบดอก ไม่ใช้คอลัมน์ Id
แบ่งแบบ stratified 80/20: train 120 แถว (คลาสละ 40), test 30 แถว (คลาสละ 10)
กำหนด random_state=42 เพื่อให้รันซ้ำได้

## ขั้นตอนที่เขียนเอง

1. ตั้งน้ำหนักตัวอย่างฝึกทุกตัวเป็น `1 / จำนวนตัวอย่าง`
2. ฝึกต้นไม้ด้วย `sample_weight` ปัจจุบัน
3. คำนวณ `error = ผลรวมน้ำหนักของตัวอย่างที่ทายผิด`
4. คำนวณน้ำหนักโมเดล `alpha = log((1-error)/error) + log(K-1)` โดย K=3
5. ตัวอย่างที่ทายผิดคูณน้ำหนักด้วย `exp(alpha)` แล้วหารน้ำหนักทุกตัวด้วยผลรวม
6. ทำซ้ำไม่เกิน 30 รอบ แล้วรวมคะแนนโหวตของต้นไม้โดยถ่วงด้วย alpha

สูตรนี้เป็น SAMME สำหรับหลายคลาส จึงมี `log(K-1)` เพิ่มจากกรณีสองคลาส
ถ้า error >= 1-1/K จะไม่รับโมเดลรอบนั้นและหยุด หากรอบแรกก็เกิดกรณีนี้จะแจ้งข้อผิดพลาด
ถ้า error=0 จะหยุดโดยเก็บโมเดลเก่าครบ สูตรปกติให้ alpha เป็นอนันต์
ในโค้ดจึงกำหนด alpha ของต้นไม้นี้เป็น `ผลรวม alpha เดิม + 1` แทน
ค่านี้ทำให้ต้นไม้ที่ทายชุดฝึกถูกทั้งหมดชนะคะแนนโหวตเดิมได้เสมอ โดยไม่ต้องใช้อนันต์
รอบนี้ไม่มีตัวอย่างที่ผิด จึงบันทึกน้ำหนักตัวอย่างเดิมไว้โดยไม่เปลี่ยน
กรณีคะแนนโหวตเสมอกัน เลือกคลาสแรกตามลำดับชื่อ

เลือก 30 รอบไว้ก่อนทดลอง ไม่เลือกจำนวนรอบจากคะแนน test
ใช้ test วัดผลหลังฝึกเสร็จเท่านั้น เปรียบเทียบกับต้นไม้เดี่ยวลึก 1 เพื่อดูผลของ Boosting

## ผลลัพธ์ใน outputs/

- `rounds.csv` — weighted error และ alpha แต่ละรอบ (error นี้มาจาก train)
- ใน `rounds.csv` มี `training_error` ของ ensemble หลังเพิ่มต้นไม้ในแต่ละรอบด้วย
- `sample_weights.csv` — แถว 0 คือน้ำหนักเริ่มต้น แถว t คือน้ำหนักหลังรอบ t; ชื่อคอลัมน์คือ Id ของข้อมูล train
- `training_error.png` — กราฟสัดส่วนที่ ensemble ทาย train ผิดในแต่ละรอบ
- `predictions.csv` — ชนิดจริงและชนิดที่ทำนาย พร้อม Id ของตัวอย่าง test
- `summary.txt` — เวลาฝึก, Accuracy, ผลรายคลาส และ Confusion Matrix

เวลาฝึกวัดด้วย `perf_counter()` เฉพาะการเรียก `fit()` รวมการคำนวณ Training Error
และเก็บประวัติภายในแต่ละรอบ ไม่รวมโหลดข้อมูล ทำนาย test หรือสร้างกราฟ
เวลาอาจต่างกันในแต่ละเครื่องและแต่ละครั้งที่รัน

ผลรันครั้งนี้: ต้นไม้เดี่ยว **66.67%** เทียบกับ Manual Boosting **90.00%** (ถูก 27/30 ตัวอย่าง)
Unit Tests ตรวจ Weighted Error, Alpha, Weight Update, Weighted Voting, ประวัติ Training Error
รวมถึงกรณีทายถูกทั้งหมดและกรณีโมเดลย่อยไม่ดีกว่าการสุ่ม โดยไม่ต้องดาวน์โหลด Dataset เพื่อรันทดสอบ

## คลาสและการตรวจแต่ละรอบ

```python
model = ManualAdaBoost(n_estimators=30)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

`models_` เก็บต้นไม้, `alphas_` เก็บน้ำหนักโหวต, `sample_weights_` เก็บสำเนาน้ำหนักตัวอย่าง
และ `history_` เก็บรอบ, weighted error, alpha, training error ตามลำดับ
ทุกครั้งที่เรียก `fit()` จะเริ่มฝึกใหม่ ไม่ต่อจากการฝึกครั้งก่อน

Weighted Error วัดต้นไม้ตัวล่าสุดโดยถ่วงด้วยน้ำหนักปัจจุบัน
ส่วน Training Error วัดผลโหวตของต้นไม้ทั้งหมดถึงรอบนั้น โดยนับตัวอย่างเท่ากัน
สองค่านี้จึงต่างกัน และ Training Error ไม่จำเป็นต้องลดลงทุก ๆ รอบ

ส่วนที่ใช้ Library: pandas โหลด/บันทึกข้อมูล, scikit-learn แบ่งข้อมูล ฝึกต้นไม้ย่อย และประเมินผล,
NumPy คำนวณอาร์เรย์, Matplotlib วาดกราฟ
ส่วนที่เขียนเองใน `ManualAdaBoost`: วนฝึกตามลำดับ คำนวณ Error/Alpha ปรับและ Normalize น้ำหนัก
เก็บประวัติ ตรวจเงื่อนไขหยุด และรวมผลด้วย Weighted Voting

Test มีเพียง 30 ตัวอย่าง ผลจึงขึ้นกับการแบ่งข้อมูล และ Boosting ไม่จำเป็นต้องดีกว่าต้นไม้เดี่ยวทุกครั้ง
อ้างอิง: [คำอธิบาย AdaBoost ของ scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
