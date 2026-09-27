"""
Stage 1 Environment & Dependency Verification Script.
Checks Python version, imports all required libraries, checks TensorFlow and Keras,
verifies CPU execution, and tests config and utils imports.
"""

import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("STAGE 1: ENVIRONMENT & DEPENDENCY VERIFICATION")
    print("=" * 60)

    # 1. Python info
    print(f"\n[1] Python Version: {sys.version}")
    major, minor = sys.version_info.major, sys.version_info.minor
    if major != 3 or minor != 12:
        print(f"    WARNING: Expected Python 3.12, got {major}.{minor}")
    else:
        print("    PASS: Python 3.12 detected.")

    # 2. Package checks
    packages = [
        ("pip", "pip"),
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "sns"),
        ("PIL", "PIL"),
        ("cv2", "cv2"),
        ("streamlit", "st"),
        ("pytest", "pytest"),
        ("tensorflow", "tf"),
    ]

    print("\n[2] Checking Package Imports:")
    failed = []
    versions = {}

    for pkg_name, alias in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            versions[pkg_name] = ver
            print(f"    - {pkg_name:<15} : OK (version: {ver})")
        except Exception as e:
            print(f"    - {pkg_name:<15} : FAILED ({e})")
            failed.append((pkg_name, str(e)))

    # 3. TensorFlow & Hardware Details
    print("\n[3] TensorFlow & Hardware Acceleration Check:")
    try:
        import tensorflow as tf
        print(f"    TensorFlow Version: {tf.__version__}")
        print(f"    Keras Version:      {tf.keras.__version__}")

        # Check physical devices
        cpus = tf.config.list_physical_devices("CPU")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"    CPUs detected: {len(cpus)}")
        print(f"    GPUs detected: {len(gpus)}")
        if gpus:
            print(f"    GPU Details: {[g.name for g in gpus]}")
        else:
            print("    Hardware Mode: Windows CPU optimized mode.")

        # Test small tensor operation
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"    Tensor test matmul shape: {c.shape} -> PASS")

    except Exception as e:
        print(f"    TensorFlow check FAILED: {e}")
        failed.append(("tensorflow_execution", str(e)))

    # 4. Project Configuration & Utils Check
    print("\n[4] Project Modules Check:")
    try:
        from src.config import config
        from src.utils import set_seed, get_logger, get_hardware_info

        set_seed(config.RANDOM_SEED)
        logger = get_logger("verification")
        hw_info = get_hardware_info()

        print(f"    Project root: {config.PROJECT_ROOT}")
        print(f"    Image size:   {config.IMAGE_SIZE}")
        print(f"    Classes:      {config.CLASS_NAMES}")
        print(f"    Hardware:     {hw_info}")
        print("    Configuration & Utils: PASS")
    except Exception as e:
        print(f"    Project modules check FAILED: {e}")
        failed.append(("project_modules", str(e)))

    print("\n" + "=" * 60)
    if failed:
        print(f"VERIFICATION STATUS: FAILED ({len(failed)} errors)")
        for item, err in failed:
            print(f"  * {item}: {err}")
        sys.exit(1)
    else:
        print("VERIFICATION STATUS: ALL CHECKS PASSED SUCCESSFULLY!")
        print("=" * 60)


if __name__ == "__main__":
    main()
