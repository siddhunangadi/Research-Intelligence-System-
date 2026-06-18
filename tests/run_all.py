"""
Master RIS Test Suite Runner.
Discovers and executes all run_tests.py scripts in subdirectories of the tests/ folder.
Executes each in an isolated subprocess to prevent state pollution.
"""

import os
import sys
import subprocess

def run_all_tests():
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tests_dir = os.path.join(workspace_root, "tests")
    
    # Locate all run_tests.py scripts in subdirectories of tests/
    test_scripts = []
    for root, dirs, files in os.walk(tests_dir):
        # Exclude root tests folder run_all.py itself
        if root == tests_dir:
            continue
        for file in files:
            if file == "run_tests.py":
                test_scripts.append(os.path.join(root, file))
                
    test_scripts.sort()
    
    print("=" * 60)
    print("RIS Master Verification Test Suite (Isolated Runs)")
    print("=" * 60)
    print(f"Found {len(test_scripts)} diagnostic test suites to run:\n")
    for script in test_scripts:
        rel_path = os.path.relpath(script, workspace_root)
        print(f"  - {rel_path}")
    print("-" * 60)

    # Environment variables override for tests
    env = os.environ.copy()
    env["DATABASE_URL"] = "sqlite:///:memory:"
    env["TEST_MODE"] = "true"

    
    success_count = 0
    failure_scripts = []
    
    for script in test_scripts:
        rel_path = os.path.relpath(script, workspace_root)
        print(f"\n>>> Running: {rel_path}...")
        
        # Execute script in isolated python process
        res = subprocess.run(
            [sys.executable, script],
            cwd=workspace_root,
            env=env,
            capture_output=False  # Allow direct streaming to terminal for active progress feedback
        )
        
        if res.returncode == 0:
            print(f"✓ {rel_path} Passed.")
            success_count += 1
        else:
            print(f"✗ {rel_path} Failed with exit code {res.returncode}.")
            failure_scripts.append(rel_path)
            
    print("\n" + "=" * 60)
    print(" RIS Test Execution Summary")
    print("=" * 60)
    print(f"Total Suites Checked : {len(test_scripts)}")
    print(f"Passed               : {success_count}")
    print(f"Failed               : {len(failure_scripts)}")
    
    if failure_scripts:
        print("\nFailed Suites:")
        for fs in failure_scripts:
            print(f"  - {fs}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\nAll diagnostics passed successfully!")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
