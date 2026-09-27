"""
Quick syntax and import test for new examples.
"""

import sys
import importlib.util

def test_import(module_name, file_path):
    """
    Test if a module can be imported without errors.

    Args:
        module_name: Name of the module
        file_path: Path to the Python file

    Returns:
        bool: True if import successful, False otherwise
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"✅ {module_name}: Import successful")
        return True
    except Exception as e:
        print(f"❌ {module_name}: Import failed - {e}")
        return False

def main():
    print("=" * 80)
    print("Testing New Examples - Syntax and Import Check")
    print("=" * 80)

    modules = [
        ("common_utils", "common_utils.py"),
        ("mmr_search", "mmr_search.py"),
        ("fusion_search", "fusion_search.py"),
        ("rag_application", "rag_application.py"),
        ("data_loader", "data_loader.py"),
    ]

    results = []
    for module_name, file_path in modules:
        success = test_import(module_name, file_path)
        results.append(success)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ All new examples passed syntax check!")
        return 0
    else:
        print("❌ Some examples failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
