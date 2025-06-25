"""
Simple Configuration for Sports Betting Alpha Mining
"""

import os

# Project Configuration
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data Sources
KAGGLE_DATASET = "austro/beat-the-bookie-worldwide-football-dataset"
STOCK_TICKER = "BVB.DE"
TARGET_TEAM = "Dortmund"

# Model Parameters
TRAIN_TEST_SPLIT = 0.7
RANDOM_STATE = 42

# Feature Columns - Only PRE-MATCH actionable features
FEATURE_COLUMNS = [
    'bvb_win_prob', 'bvb_opponent_prob', 'draw_prob', 'bookmaker_margin', 
    'is_bundesliga', 'is_champions_league', 'is_europa_league', 'bvb_home'
]

# POST-MATCH features (not actionable for prediction)
POST_MATCH_FEATURES = [
    'bvb_won', 'surprise_factor', 'total_goals', 'goal_difference'
]

# Plot Settings
FIGURE_SIZE = (12, 8)
