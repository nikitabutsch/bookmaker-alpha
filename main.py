"""
Simple Sports Betting Alpha Mining - Proof of Concept
"""

import pandas as pd
import numpy as np
import config
import os

# Import our modules
from data_loader import StockDataLoader
from analysis import AlphaSignalAnalyzer

def main():
    """Main pipeline execution"""
    
    print("🚀 Sports Betting Alpha Mining - Proof of Concept")
    print("📖 Finding alpha signals from BVB matches to predict stock returns\n")
    
    # =================================================================
    # 1. LOAD EXISTING ALPHA DATASET
    # =================================================================
    dataset_path = f"{config.RESULTS_DIR}/alpha_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print("❌ No existing alpha dataset found!")
        print("Please run the full data collection pipeline first.")
        return
    
    alpha_dataset = pd.read_csv(dataset_path)
    alpha_dataset['match_date'] = pd.to_datetime(alpha_dataset['match_date'])
    
    print(f"📊 Loaded {len(alpha_dataset)} BVB matches ({alpha_dataset['match_date'].min().date()} to {alpha_dataset['match_date'].max().date()})")
    
    # =================================================================
    # 2. ANALYZE ALPHA SIGNALS
    # =================================================================
    analyzer = AlphaSignalAnalyzer()
    
    # Generate a high-level overview of the dataset
    print("\n📋 Generating dataset overview...")
    analyzer.generate_dataset_overview(alpha_dataset)
    
    # Run the detailed alpha signal analysis
    print("\n🔍 Analyzing predictive alpha signals...")
    analysis_results = analyzer.analyze_alpha_signals(alpha_dataset)
    
    # =================================================================
    # 3. SUMMARY
    # =================================================================
    print("\n✅ Analysis complete!")
    
    return alpha_dataset, analysis_results

if __name__ == "__main__":
    main()
