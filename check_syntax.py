#!/usr/bin/env python3
"""
Quick syntax check for all implementation files.

This checks Python syntax without requiring heavy dependencies.
"""

import py_compile
import sys
from pathlib import Path

def check_file(filepath):
    """Check if a Python file has valid syntax."""
    try:
        py_compile.compile(str(filepath), doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Check all implementation files."""
    project_dir = Path(__file__).parent
    
    files_to_check = [
        'ocr_adapter.py',
        'semantic_reconciler.py',
        'vlm_backend.py',
        'finetune_qwen.py',
        'production_monitor.py',
        'validate_implementation.py',
        'test_phase1.py',
        'config.py',
        'ocr_extractor.py',
    ]
    
    print("=" * 60)
    print("SYNTAX CHECK FOR IMPLEMENTATION FILES")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    for filename in files_to_check:
        filepath = project_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  SKIP: {filename} (file not found)")
            continue
        
        success, error = check_file(filepath)
        
        if success:
            print(f"✅ PASS: {filename}")
            passed += 1
        else:
            print(f"❌ FAIL: {filename}")
            if error:
                print(f"   Error: {error[:100]}...")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ All files have valid Python syntax!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run validation: python3 validate_implementation.py")
        print("  3. Read guides: INSTALLATION.md and QUICK_START.md")
        return 0
    else:
        print("\n❌ Some files have syntax errors. Fix them above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
