#!/usr/bin/env python3
"""
Simple script to run the CLEAR-E dataset preprocessing
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_datasets import DatasetPreprocessor

def main():
    """Run preprocessing with error handling"""
    try:
        print("Starting CLEAR-E Dataset Preprocessing...")
        print("=" * 60)
        
        # Initialize preprocessor
        preprocessor = DatasetPreprocessor()
        
        # Run preprocessing
        datasets = preprocessor.preprocess_all_datasets()
        
        print("\n✅ Preprocessing completed successfully!")
        print(f"📁 Processed datasets available in: {preprocessor.output_dir}/")
        
        # List output files
        output_files = os.listdir(preprocessor.output_dir)
        if output_files:
            print("\n📋 Generated files:")
            for file in sorted(output_files):
                file_path = os.path.join(preprocessor.output_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  • {file} ({size_mb:.2f} MB)")
        
        return datasets
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that all dataset files are present:")
        print("   - ECL_data.zip")
        print("   - GEFCom2014_data.zip") 
        print("   - SouthernChina_data/Transformer_DB.db")
        print("2. Ensure you have required Python packages:")
        print("   pip install pandas numpy sqlite3")
        print("3. Check file permissions and disk space")
        return None

if __name__ == "__main__":
    datasets = main()
