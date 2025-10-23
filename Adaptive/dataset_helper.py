#!/usr/bin/env python3
"""
Dataset Path Helper for Adaptive NeRD-Rain Training

This script helps identify the correct dataset path for different environments.
"""

import os
import sys
import argparse

def find_dataset_path():
    """Find the dataset path in common locations"""
    common_paths = [
        './Datasets',                    # Local development
        '../Datasets',                   # One level up
        '../../Datasets',               # Two levels up
        '/content/Datasets',            # Google Colab
        '/content/drive/MyDrive/Datasets',  # Google Drive in Colab
        './data/Datasets',              # Alternative data folder
        '~/Datasets',                   # Home directory
        '/kaggle/input',                # Kaggle
    ]
    
    found_paths = []
    
    for path in common_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            # Check if it has the expected structure
            rain200l_path = os.path.join(expanded_path, 'Rain200L')
            if os.path.exists(rain200l_path):
                train_path = os.path.join(rain200l_path, 'train')
                if os.path.exists(train_path):
                    found_paths.append(os.path.abspath(expanded_path))
    
    return found_paths

def check_dataset_structure(data_path):
    """Check if the dataset has the expected structure"""
    required_dirs = [
        'Rain200L/train/input',
        'Rain200L/train/target',
        'Rain200L/test/input',
        'Rain200L/test/target'
    ]
    
    missing_dirs = []
    for req_dir in required_dirs:
        full_path = os.path.join(data_path, req_dir)
        if not os.path.exists(full_path):
            missing_dirs.append(req_dir)
    
    return missing_dirs

def main():
    parser = argparse.ArgumentParser(description='Dataset Path Helper')
    parser.add_argument('--check-path', type=str, help='Check specific path')
    parser.add_argument('--find-datasets', action='store_true', help='Find dataset paths')
    
    args = parser.parse_args()
    
    print("=== Adaptive NeRD-Rain Dataset Path Helper ===")
    print()
    
    if args.check_path:
        print(f"Checking path: {args.check_path}")
        if os.path.exists(args.check_path):
            print("✅ Path exists")
            missing = check_dataset_structure(args.check_path)
            if not missing:
                print("✅ Dataset structure is complete")
                print(f"✅ Use this path: {os.path.abspath(args.check_path)}")
            else:
                print("❌ Missing directories:")
                for missing_dir in missing:
                    print(f"   - {missing_dir}")
        else:
            print("❌ Path does not exist")
        return
    
    if args.find_datasets:
        print("Searching for datasets in common locations...")
        found_paths = find_dataset_path()
        
        if found_paths:
            print(f"✅ Found {len(found_paths)} valid dataset(s):")
            for i, path in enumerate(found_paths, 1):
                print(f"   {i}. {path}")
                missing = check_dataset_structure(path)
                if missing:
                    print(f"      ⚠️  Missing: {', '.join(missing)}")
                else:
                    print(f"      ✅ Complete dataset structure")
            
            print()
            print("Usage examples:")
            print(f"  python Adaptive/train_adaptive.py --data_path '{found_paths[0]}'")
            if len(found_paths) > 1:
                print(f"  python Adaptive/train_adaptive.py --data_path '{found_paths[1]}'")
        else:
            print("❌ No valid datasets found in common locations")
            print()
            print("Please ensure your dataset follows this structure:")
            print("  Datasets/")
            print("    Rain200L/")
            print("      train/")
            print("        input/    (rainy images)")
            print("        target/   (clean images)")
            print("      test/")
            print("        input/    (rainy images)")
            print("        target/   (clean images)")
        return
    
    # Default: show current environment info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Check if default path exists
    default_path = './Datasets'
    if os.path.exists(default_path):
        print(f"✅ Default dataset path exists: {os.path.abspath(default_path)}")
        missing = check_dataset_structure(default_path)
        if not missing:
            print("✅ Dataset structure is complete")
        else:
            print("❌ Dataset structure incomplete. Missing:")
            for missing_dir in missing:
                print(f"   - {missing_dir}")
    else:
        print(f"❌ Default dataset path not found: {os.path.abspath(default_path)}")
    
    print()
    print("Options:")
    print("  --find-datasets    : Search for datasets in common locations")
    print("  --check-path PATH  : Check if a specific path is valid")

if __name__ == '__main__':
    main()