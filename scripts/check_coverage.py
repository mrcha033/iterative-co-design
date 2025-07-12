#!/usr/bin/env python3
"""
Check code coverage and ensure it meets minimum requirements.

This script runs tests with coverage analysis and validates that coverage
meets the required threshold for the project.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def run_tests_with_coverage(test_dir="tests/unit", min_coverage=90):
    """
    Run tests with coverage analysis.
    
    Args:
        test_dir: Directory containing tests to run
        min_coverage: Minimum coverage percentage required
        
    Returns:
        bool: True if coverage meets minimum requirement
    """
    print(f"Running tests in {test_dir} with coverage analysis...")
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        test_dir,
        "--cov=src",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "--cov-report=html",
        f"--cov-fail-under={min_coverage}",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check if coverage XML file was generated
        coverage_file = Path("coverage.xml")
        if coverage_file.exists():
            coverage_pct = parse_coverage_xml(coverage_file)
            print(f"\nOverall coverage: {coverage_pct:.1f}%")
            
            if coverage_pct >= min_coverage:
                print(f"✅ Coverage {coverage_pct:.1f}% meets minimum requirement of {min_coverage}%")
                return True
            else:
                print(f"❌ Coverage {coverage_pct:.1f}% is below minimum requirement of {min_coverage}%")
                return False
        else:
            print("❌ Coverage XML file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Test execution failed: {e}")
        return False


def parse_coverage_xml(coverage_file):
    """
    Parse coverage XML file to extract overall coverage percentage.
    
    Args:
        coverage_file: Path to coverage.xml file
        
    Returns:
        float: Coverage percentage
    """
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        # Find coverage element
        for coverage in root.iter('coverage'):
            line_rate = coverage.get('line-rate')
            if line_rate:
                return float(line_rate) * 100
        
        # Alternative: look for package-level coverage
        for package in root.iter('package'):
            line_rate = package.get('line-rate')
            if line_rate:
                return float(line_rate) * 100
                
        return 0.0
        
    except Exception as e:
        print(f"Error parsing coverage XML: {e}")
        return 0.0


def check_missing_coverage():
    """
    Analyze which files have low coverage and need attention.
    
    Returns:
        List of files with coverage issues
    """
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        return []
    
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        low_coverage_files = []
        
        for class_elem in root.iter('class'):
            filename = class_elem.get('filename', '')
            line_rate = float(class_elem.get('line-rate', '0'))
            
            if line_rate < 0.8:  # Less than 80% coverage
                low_coverage_files.append({
                    'file': filename,
                    'coverage': line_rate * 100
                })
        
        return sorted(low_coverage_files, key=lambda x: x['coverage'])
        
    except Exception as e:
        print(f"Error analyzing coverage: {e}")
        return []


def generate_coverage_report():
    """Generate detailed coverage report."""
    print("\n" + "="*60)
    print("DETAILED COVERAGE ANALYSIS")
    print("="*60)
    
    low_coverage = check_missing_coverage()
    
    if low_coverage:
        print("\nFiles with low coverage (< 80%):")
        for item in low_coverage:
            print(f"  {item['file']}: {item['coverage']:.1f}%")
        
        print(f"\nTotal files needing attention: {len(low_coverage)}")
        
        print("\nRecommendations:")
        print("1. Add unit tests for uncovered functions")
        print("2. Test error handling and edge cases")
        print("3. Mock external dependencies in tests")
        print("4. Consider integration tests for complex workflows")
    else:
        print("\n✅ All files have good coverage (≥ 80%)")


def run_deterministic_tests():
    """
    Run tests multiple times to check for deterministic behavior.
    
    Returns:
        bool: True if tests are deterministic
    """
    print("\nChecking test determinism...")
    
    results = []
    for run in range(3):
        print(f"Test run {run + 1}/3...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "--tb=no",
            "-q",
            "--seed=42"  # Fixed seed for deterministic tests
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            results.append(result.returncode)
        except Exception as e:
            print(f"Error in test run {run + 1}: {e}")
            return False
    
    # All runs should have same result
    if all(r == results[0] for r in results):
        print("✅ Tests are deterministic")
        return True
    else:
        print("❌ Tests show non-deterministic behavior")
        print(f"Return codes: {results}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Check code coverage and test quality')
    parser.add_argument('--min-coverage', type=float, default=90.0,
                       help='Minimum coverage percentage required (default: 90.0)')
    parser.add_argument('--test-dir', type=str, default='tests/unit',
                       help='Directory containing tests to run (default: tests/unit)')
    parser.add_argument('--check-determinism', action='store_true',
                       help='Check if tests are deterministic')
    parser.add_argument('--detailed-report', action='store_true',
                       help='Generate detailed coverage report')
    
    args = parser.parse_args()
    
    print("Code Coverage and Quality Check")
    print("="*40)
    
    success = True
    
    # Run coverage analysis
    if not run_tests_with_coverage(args.test_dir, args.min_coverage):
        success = False
    
    # Generate detailed report if requested
    if args.detailed_report:
        generate_coverage_report()
    
    # Check test determinism if requested
    if args.check_determinism:
        if not run_deterministic_tests():
            success = False
    
    print("\n" + "="*40)
    if success:
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()