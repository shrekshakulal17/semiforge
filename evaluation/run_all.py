import os
import sys
import subprocess


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def run_script(script_name):

    script_path = os.path.join(
        PROJECT_ROOT,
        "evaluation",
        script_name
    )

    print()
    print("=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            script_path
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print()
        print(f"❌ {script_name} failed.")

        return False

    print()
    print(f"✅ {script_name} completed.")

    return True


def main():

    print("=" * 60)
    print("KLA MODEL EVALUATION PIPELINE")
    print("=" * 60)

    scripts = [
        "evaluate.py",
        "visual_test.py",
        "lpips_test.py",
        "benchmark.py"
    ]

    for script in scripts:

        success = run_script(script)

        if not success:

            print()
            print("Evaluation stopped.")
            sys.exit(1)


    print()
    print("=" * 60)
    print("ALL EVALUATION TESTS COMPLETED")
    print("=" * 60)

    print()
    print("Generated results:")
    print("• PSNR")
    print("• SSIM")
    print("• LPIPS")
    print("• Inference time")
    print("• Visual restoration images")


if __name__ == "__main__":
    main()
    