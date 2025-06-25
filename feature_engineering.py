"""
Feature engineering for alpha signals from betting odds
"""

import pandas as pd
import numpy as np
import config

class AlphaFeatureEngineer:
    """Engineer features for alpha signal extraction"""
    
    def __init__(self):
        self.features_df = None
    
    def _get_next_trading_day(self, match_date, stock_data):
        """Find the next available trading day after match date (match date can be on weekend)"""
        next_dates = stock_data.index[stock_data.index > match_date]
        return next_dates[0] if len(next_dates) > 0 else None
    
    def _normalize_probabilities(self, home_prob, draw_prob, away_prob):
        """Normalize probabilities to remove bookmaker margin"""
        total_prob = home_prob + draw_prob + away_prob
        if pd.notna(total_prob) and total_prob > 0:
            margin = total_prob - 1
            return (home_prob / total_prob, draw_prob / total_prob,
                    away_prob / total_prob, margin)
        return np.nan, np.nan, np.nan, np.nan
    
    def _calculate_surprise_factor(self, match_outcome, bvb_won, bvb_win_prob, draw_prob, opponent_prob):
        """Calculate surprise factor based on outcome vs expectations"""
        if pd.isna(bvb_win_prob):
            return np.nan
        
        if bvb_won == 1:
            return 1 - bvb_win_prob  # Higher if unexpected win
        elif match_outcome == 0:  # Draw
            return 1 - draw_prob if pd.notna(draw_prob) else np.nan
        else:  # Loss
            return 1 - opponent_prob if pd.notna(opponent_prob) else np.nan
    
    def process_matches(self, matches_df, stock_data):
        """Process matches and extract alpha features"""
        features_list = []
        
        for idx, match in matches_df.iterrows():
            try:
                match_date = pd.to_datetime(match['match_date'])
                
                # Find next trading day
                next_trading_day = self._get_next_trading_day(match_date, stock_data)
                if next_trading_day is None:
                    continue
                
                # Get target returns
                try:
                    next_day_return = stock_data.loc[next_trading_day, 'Daily_Return']
                    three_day_return = stock_data.loc[next_trading_day, 'Next_3Day_Return']
                except KeyError:
                    continue
                
                # Extract betting features
                home_prob = 1 / match['avg_odds_home_win'] if pd.notna(match['avg_odds_home_win']) else np.nan
                draw_prob = 1 / match['avg_odds_draw'] if pd.notna(match['avg_odds_draw']) else np.nan
                away_prob = 1 / match['avg_odds_away_win'] if pd.notna(match['avg_odds_away_win']) else np.nan
                
                # Normalize probabilities (remove bookmaker margin)
                home_prob_norm, draw_prob_norm, away_prob_norm, margin = self._normalize_probabilities(
                    home_prob, draw_prob, away_prob
                )
                
                # Match outcome analysis
                if match['home_score'] > match['away_score']:
                    match_outcome = 1  # Home win
                elif match['home_score'] < match['away_score']:
                    match_outcome = -1  # Away win
                else:
                    match_outcome = 0  # Draw
                
                # BVB specific features
                bvb_home = 1 if config.TARGET_TEAM.lower() in match['home_team'].lower() else 0
                bvb_away = 1 if config.TARGET_TEAM.lower() in match['away_team'].lower() else 0
                
                if bvb_home:
                    bvb_won = 1 if match_outcome == 1 else 0
                    bvb_win_prob = home_prob_norm
                    opponent_prob = away_prob_norm
                else:
                    bvb_won = 1 if match_outcome == -1 else 0
                    bvb_win_prob = away_prob_norm
                    opponent_prob = home_prob_norm
                
                # Calculate surprise factor
                surprise_factor = self._calculate_surprise_factor(
                    match_outcome, bvb_won, bvb_win_prob, draw_prob_norm, opponent_prob
                )
                
                # Match importance features
                league_features = self._extract_league_features(match['league'])
                
                # Compile features
                features = {
                    'match_id': match['match_id'],
                    'match_date': match_date,
                    'next_trading_day': next_trading_day,
                    'next_day_return': next_day_return,
                    'three_day_return': three_day_return,
                    'stock_up_next_day': 1 if next_day_return > 0 else 0,
                    'bvb_home': bvb_home,
                    'bvb_away': bvb_away,
                    'bvb_won': bvb_won,
                    'match_outcome': match_outcome,
                    'bvb_win_prob': bvb_win_prob,
                    'opponent_prob': opponent_prob,
                    'draw_prob': draw_prob_norm,
                    'bookmaker_margin': margin,
                    'surprise_factor': surprise_factor,
                    'total_goals': match['home_score'] + match['away_score'],
                    'goal_difference': abs(match['home_score'] - match['away_score']),
                    **league_features
                }
                
                features_list.append(features)
                
            except Exception as e:
                print(f"Error processing match {match.get('match_id', 'unknown')}: {e}")
                continue
        
        self.features_df = pd.DataFrame(features_list)
        return self.features_df
    
    def _extract_league_features(self, league):
        """Extract league features"""
        league_lower = league.lower()
        return {
            'is_bundesliga': 1 if 'bundesliga' in league_lower and '2.' not in league_lower else 0,
            'is_champions_league': 1 if 'champions league' in league_lower else 0,
            'is_europa_league': 1 if 'europa league' in league_lower or 'uefa cup' in league_lower else 0,
            'is_domestic_cup': 1 if 'dfb pokal' in league_lower else 0,
            'is_friendly': 1 if 'friendly' in league_lower else 0
        }
    
    def get_feature_dataset(self):
        """Get the engineered feature dataset"""
        return self.features_df
    
    def save_features(self, filepath):
        """Save features to CSV"""
        if self.features_df is not None:
            self.features_df.to_csv(filepath, index=False)
            print(f"Features saved to {filepath}")
        else:
            print("No features to save. Run process_matches first.")


def main():
    """Test feature engineering"""
    from data_loader import BettingDataLoader, StockDataLoader
    
    # Load data
    betting_loader = BettingDataLoader()
    stock_loader = StockDataLoader()
    
    data_path = betting_loader.download_data()
    betting_loader.load_data(data_path)
    matches = betting_loader.filter_team_matches()
    
    stock_data = stock_loader.download_data()
    
    # Engineer features
    engineer = AlphaFeatureEngineer()
    features = engineer.process_matches(matches, stock_data)

    # Save engineered dataset for downstream analysis
    output_path = f"{config.RESULTS_DIR}/alpha_dataset.csv"
    engineer.save_features(output_path)

    print(f"Engineered {len(features)} match features")
    print(f"Feature columns: {list(features.columns)}")
    print(f"✅ Alpha dataset stored at {output_path}")

if __name__ == "__main__":
    main() 