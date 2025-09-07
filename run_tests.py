#!/usr/bin/env python3
"""
Test runner script for Advanced Document Search
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Run tests for Advanced Document Search')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only') 
    parser.add_argument('--api', action='store_true', help='Run API tests only')
    parser.add_argument('--slow', action='store_true', help='Include slow tests')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fast', action='store_true', help='Run fast tests only (exclude slow tests)')
    parser.add_argument('--file', type=str, help='Run specific test file')
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    original_cwd = Path.cwd()
    
    try:
        import os
        os.chdir(project_dir)
        
        # Build pytest command
        cmd = [sys.executable, '-m', 'pytest']
        
        # Add coverage if requested
        if args.coverage:
            cmd.extend(['--cov=backend', '--cov-report=html', '--cov-report=term'])
        
        # Add verbosity
        if args.verbose:
            cmd.append('-v')
        
        # Filter by test type
        if args.unit:
            cmd.extend(['-m', 'unit'])
        elif args.integration:
            cmd.extend(['-m', 'integration'])
        elif args.api:
            cmd.extend(['-m', 'api'])
        elif args.fast:
            cmd.extend(['-m', 'not slow'])
        elif not args.slow:
            cmd.extend(['-m', 'not slow'])
        
        # Specific file
        if args.file:
            cmd.append(f'tests/{args.file}')
        
        # Run the tests
        success = run_command(cmd, "Running pytest")
        
        if args.coverage and success:
            print("\n" + "="*60)
            print("Coverage report generated in htmlcov/index.html")
            print("="*60)
        
        return 0 if success else 1
        
    finally:
        os.chdir(original_cwd)

if __name__ == '__main__':
    sys.exit(main())