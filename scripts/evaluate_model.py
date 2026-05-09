import argparse
import json
import sys
import os


def evaluate_model(metrics_file: str, accuracy_threshold: float, f1_threshold: float) -> bool:
    if not os.path.exists(metrics_file):
        print(f"❌ ERROR: File {metrics_file} tidak ditemukan!")
        print("   Pastikan train.py menghasilkan metrics.json")
        return False

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy", 0.0)
    f1_score = metrics.get("f1_score", 0.0)

    print("=" * 60)
    print("📊 MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy   : {accuracy:.4f}  (threshold: {accuracy_threshold})")
    print(f"  F1-Score   : {f1_score:.4f}  (threshold: {f1_threshold})")
    print("=" * 60)

    # Validasi terhadap threshold
    accuracy_passed = accuracy >= accuracy_threshold
    f1_passed = f1_score >= f1_threshold

    print(f"  Accuracy Check : {'✅ PASS' if accuracy_passed else '❌ FAIL'}")
    print(f"  F1-Score Check : {'✅ PASS' if f1_passed else '❌ FAIL'}")
    print("=" * 60)

    all_passed = accuracy_passed and f1_passed

    if all_passed:
        print("🎉 MODEL VALIDATION: PASSED")
        print("   Model layak naik ke tahap registry.")
    else:
        print("⚠️  MODEL VALIDATION: FAILED")
        print("   Model TIDAK akan didaftarkan ke registry.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model terhadap threshold")
    parser.add_argument("--metrics-file", default="metrics.json",
                        help="Path ke file metrics.json")
    parser.add_argument("--accuracy-threshold", type=float, default=0.75,
                        help="Threshold minimum akurasi")
    parser.add_argument("--f1-threshold", type=float, default=0.70,
                        help="Threshold minimum F1-score")
    args = parser.parse_args()

    passed = evaluate_model(
        metrics_file=args.metrics_file,
        accuracy_threshold=args.accuracy_threshold,
        f1_threshold=args.f1_threshold,
    )

    # Exit code 0 = success, 1 = failure (untuk CI/CD)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()