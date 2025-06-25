"""
Simple data loader for betting data and stock data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime

import config

# Note: importing Kaggle triggers an automatic authentication attempt.
# We therefore postpone the import until we actually need it (inside download_data)

class BettingDataLoader:
    """Simple class to download and load betting data"""
    
    def __init__(self):
        self.raw_data = None
        self.team_matches = None
    
    def download_data(self):
        """Download betting data from Kaggle – fallback to local copy on any error."""

        def _local_dataset_available() -> bool:
            """Check whether the expected betting data file already exists in DATA_DIR."""
            return any(
                fname.lower().endswith((".csv", ".gz")) and "closing_odds" in fname.lower()
                for fname in os.listdir(config.DATA_DIR)
            )

        # If the dataset is already there, short-circuit immediately
        if _local_dataset_available():
            print("✅ Found local betting dataset – skipping Kaggle download.")
            return config.DATA_DIR

        # Attempt to grab the dataset from Kaggle
        try:
            print(
                f"⬇️  Downloading Kaggle dataset '{config.KAGGLE_DATASET}' – this can take a few minutes…"
            )
            import importlib
            kaggle = importlib.import_module("kaggle")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                config.KAGGLE_DATASET,
                path=config.DATA_DIR,
                unzip=True,
                quiet=False,
            )
            print("✅ Dataset downloaded and extracted to", config.DATA_DIR)
            return config.DATA_DIR

        except Exception as kaggle_err:
            # Something went wrong (credentials, rate-limit, network, licence not accepted…)
            print(
                f"⚠️  Kaggle download failed: {kaggle_err}\n"
                "   Attempting to fall back to a local copy in the 'data/' directory…"
            )

            if _local_dataset_available():
                print("✅ Local copy found – continuing with existing files.")
                return config.DATA_DIR

            # Nothing to fall back to: raise a clear error
            raise RuntimeError(
                "Unable to retrieve the betting dataset. Kaggle download failed and no local "
                "'closing_odds' (CSV/GZ) file was found in the data directory. Please either:\n"
                "  • Place the dataset files in the 'data/' folder manually, or\n"
                "  • Configure Kaggle credentials and re-run the script."
            ) from kaggle_err
    
    def load_data(self, data_path):
        """Load raw betting data"""
        # Look for the main data file
        for file in os.listdir(data_path):
            if 'closing_odds' in file.lower():
                file_path = os.path.join(data_path, file)
                
                if file.endswith('.gz'):
                    self.raw_data = pd.read_csv(file_path, compression='gzip')
                else:
                    self.raw_data = pd.read_csv(file_path)
                
                # Convert date column
                self.raw_data['match_date'] = pd.to_datetime(self.raw_data['match_date'])
                
                return self.raw_data
        
        raise FileNotFoundError("Could not find betting data file")
    
    def filter_team_matches(self, data=None, team_name=None):
        """Filter matches for a specific team"""
        if data is None:
            data = self.raw_data
        if team_name is None:
            team_name = config.TARGET_TEAM
            
        # Filter for team matches (home or away)
        team_home = data[data['home_team'].str.contains(team_name, case=False, na=False)]
        team_away = data[data['away_team'].str.contains(team_name, case=False, na=False)]
        team_matches = pd.concat([team_home, team_away]).drop_duplicates()
        
        # Keep only matches with valid odds
        team_matches = team_matches.dropna(subset=['avg_odds_home_win', 'avg_odds_draw', 'avg_odds_away_win'])
        
        # Sort by date
        team_matches = team_matches.sort_values('match_date').reset_index(drop=True)
        
        return team_matches


class StockDataLoader:
    """Simple class to download and process stock data"""
    
    def __init__(self):
        self.stock_data = None
    
    def download_data(self, start_date=None, end_date=None):
        """Download stock data from Yahoo Finance"""
        if start_date is None:
            start_date = "2005-01-01"
        if end_date is None:
            end_date = datetime.now()
        
        # Download stock data
        self.stock_data = yf.download(config.STOCK_TICKER, start=start_date, end=end_date)
        
        # Handle potential MultiIndex columns
        if isinstance(self.stock_data.columns, pd.MultiIndex):
            self.stock_data.columns = self.stock_data.columns.droplevel(1)
        
        # Use Close price if Adj Close not available
        if 'Adj Close' not in self.stock_data.columns:
            if 'Close' in self.stock_data.columns:
                self.stock_data['Adj Close'] = self.stock_data['Close']
        
        # Calculate returns directly using pct_change
        self.stock_data['Daily_Return'] = self.stock_data['Adj Close'].pct_change()
        self.stock_data['Next_Day_Return'] = self.stock_data['Daily_Return'].shift(-1)
        self.stock_data['Next_3Day_Return'] = self.stock_data['Adj Close'].pct_change(periods=3).shift(-3)
        
        return self.stock_data
