"""วาดกราฟ Accuracy เวลาฝึก และเวลา test จากผลใน model_comparison.json"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")  # บันทึกรูปได้โดยไม่ต้องเปิดหน้าต่าง GUI
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter


RESULT_PATH = ROOT / "results" / "model_comparison.json"
OUTPUT_PATH = ROOT / "results" / "accuracy_time_comparison"
TEST_OUTPUT_PATH = ROOT / "results" / "test_time_comparison"
TIME_SCALE = "log"  # ใช้ log เพราะเวลาของแต่ละวิธีต่างกันมาก; เปลี่ยนเป็น linear ได้


def plot_test_time(rows, labels, colors):
    """เปรียบเทียบเวลาทำนาย test ทั้งชุด โดยแสดงหน่วยมิลลิวินาที"""
    test_times_ms = [row["test_seconds"] * 1000 for row in rows]
    positions = range(len(rows))

    figure, axis = plt.subplots(figsize=(9, 6))
    figure.subplots_adjust(left=0.12, right=0.96, bottom=0.25, top=0.76)
    figure.suptitle("Iris: test time", x=0.12, y=0.95, ha="left", fontsize=23)
    figure.text(
        0.12, 0.88,
        f"Same {rows[0]['test_samples']} test samples | Lower is better",
        fontsize=12,
    )

    bars = axis.bar(positions, test_times_ms, width=0.54, color=colors)
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("Prediction time (milliseconds)")
    axis.set_ylim(0, max(test_times_ms) * 1.25)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E3E8EE", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#CBD3DD")
    axis.tick_params(axis="both", length=0, pad=9)
    axis.bar_label(bars, labels=[f"{time_ms:.3f} ms" for time_ms in test_times_ms], padding=8)

    figure.text(0.12, 0.10, "Includes scaling, prediction and ensemble voting.", fontsize=11)
    figure.text(
        0.12, 0.055,
        "One measured run. Excludes training and accuracy calculation; times vary.",
        fontsize=10, color="#526070",
    )
    for extension in ("png", "svg"):
        output_file = TEST_OUTPUT_PATH.with_suffix(f".{extension}")
        figure.savefig(output_file, dpi=200, facecolor="white")
        print(f"Saved: {output_file}")
    plt.close(figure)


def main():
    # 1. อ่านผลเดิม แล้วเรียง Bagging → Boosting → SVM
    report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    results_by_name = {row["method"]: row for row in report["results"]}
    methods = ["Bagging (4 types)", "Boosting (4 types)", "Single SVM"]
    rows = [results_by_name[method] for method in methods]
    names = ["Bagging", "Boosting", "SVM"]
    labels = [f"{name}\n({row['model_count']} models)" for name, row in zip(names, rows)]
    labels[-1] = "SVM\n(1 model)"
    accuracies = [row["accuracy"] * 100 for row in rows]
    training_times = [row["training_seconds"] for row in rows]
    colors = ["#3576C4", "#169A83", "#E5A23B"]
    positions = range(len(rows))

    # 2. สร้างสองกราฟ เพื่อแยกหน่วยเปอร์เซ็นต์กับวินาที
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelcolor": "#263244",
        "text.color": "#263244",
        "xtick.color": "#263244",
        "ytick.color": "#526070",
        "svg.fonttype": "none",
    })
    figure, (accuracy_axis, time_axis) = plt.subplots(1, 2, figsize=(12.5, 6.5))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.25, top=0.77, wspace=0.30)
    figure.suptitle("Iris: accuracy & training time", x=0.08, y=0.96, ha="left", fontsize=23)
    figure.text(
        0.08, 0.89,
        f"Same split: {report['train_samples']} train / {report['test_samples']} test"
        f"  |  Seed {report['seed']}",
        fontsize=12,
    )

    for axis in (accuracy_axis, time_axis):
        axis.set_xticks(list(positions), labels)
        axis.set_xlim(-0.65, 2.65)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E3E8EE", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#CBD3DD")
        axis.tick_params(axis="both", length=0, pad=9)

    # 3. Accuracy: กราฟแท่งเริ่มจากศูนย์ พร้อมค่าที่วัดได้จริง
    accuracy_axis.bar(positions, accuracies, width=0.54, color=colors)
    accuracy_axis.set_title("Accuracy — higher is better", loc="left", pad=22)
    accuracy_axis.set_ylabel("Test accuracy (%)")
    accuracy_axis.set_ylim(0, 110)
    accuracy_axis.set_yticks([0, 20, 40, 60, 80, 100])
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    for position, accuracy in zip(positions, accuracies):
        accuracy_axis.text(position, accuracy + 2, f"{accuracy:.2f}%", ha="center", fontsize=13)

    # 4. เวลา: กราฟจุดบนแกน log ช่วยให้เห็นวิธีที่ใช้เวลาน้อยด้วย
    time_axis.scatter(positions, training_times, s=155, c=colors, zorder=3)
    time_axis.set_title("Training time — lower is better", loc="left", pad=22)
    time_axis.set_yscale(TIME_SCALE)
    time_axis.set_ylabel(f"Seconds ({TIME_SCALE} scale)")
    if TIME_SCALE == "log":
        time_axis.set_ylim(min(training_times) / 4, max(training_times) * 4)
        time_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        time_axis.minorticks_off()
    else:
        time_axis.set_ylim(0, max(training_times) * 1.3)
    for position, seconds in zip(positions, training_times):
        time_axis.annotate(
            f"{seconds:.4f} s", (position, seconds),
            xytext=(0, 13), textcoords="offset points", ha="center", fontsize=13,
        )

    figure.text(
        0.08, 0.10,
        "Bagging / Boosting: SVM + KNN + Gaussian Naive Bayes + Logistic Regression.",
        fontsize=11,
    )
    figure.text(
        0.08, 0.055,
        "One recorded run. Bagging time includes worker startup; times vary between runs.",
        fontsize=11, color="#526070",
    )

    # 5. บันทึกรูป PNG และไฟล์ SVG ที่ขยายได้โดยภาพไม่แตก
    for extension in ("png", "svg"):
        output_file = OUTPUT_PATH.with_suffix(f".{extension}")
        figure.savefig(output_file, dpi=200, facecolor="white")
        print(f"Saved: {output_file}")
    plt.close(figure)

    # 6. วาดเวลา test แยกต่างหาก เพราะใช้หน่วยมิลลิวินาทีและแกนปกติ
    plot_test_time(rows, labels, colors)


if __name__ == "__main__":
    main()
