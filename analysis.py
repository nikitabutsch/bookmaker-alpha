"""
Simple alpha signal analysis
"""

import pandas as pd
import numpy as np
from scipy import stats

class AlphaSignalAnalyzer:
    """Simple class to analyze alpha signals"""
    
    def __init__(self):
        pass
    
    def generate_dataset_overview(self, alpha_dataset):
        """Basic dataset overview for alpha mining context"""
        # Clean data - remove rows with missing returns
        clean_data = alpha_dataset.dropna(subset=['next_day_return'])
        
        # Dataset composition for alpha mining
        total_matches = len(clean_data)
        bundesliga_matches = len(clean_data[clean_data['is_bundesliga'] == 1])
        champions_league_matches = len(clean_data[clean_data['is_champions_league'] == 1])
        europa_league_matches = len(clean_data[clean_data['is_europa_league'] == 1])
        
        # Basic return statistics
        mean_return = clean_data['next_day_return'].mean()
        volatility = clean_data['next_day_return'].std()
        positive_days = (clean_data['next_day_return'] > 0).mean()
        
        # Data quality for alpha mining
        complete_odds_data = len(clean_data.dropna(subset=['bvb_win_prob', 'bookmaker_margin']))
        
        # Average next-day return conditioned on match outcome
        wins = clean_data[clean_data['bvb_won'] == 1]['next_day_return']
        losses = clean_data[clean_data['bvb_won'] == 0]['next_day_return']
        win_return = wins.mean()
        loss_return = losses.mean()
        
        # Print dataset overview
        print("\n" + "="*50)
        print("DATASET OVERVIEW FOR ALPHA MINING")
        print("="*50)
        
        print(f"Total matches: {total_matches}")
        print(f"Complete odds data: {complete_odds_data} ({complete_odds_data/total_matches:.1%})")
        print(f"\nMatch composition:")
        print(f"  Bundesliga: {bundesliga_matches} ({bundesliga_matches/total_matches:.1%})")
        print(f"  Champions League: {champions_league_matches} ({champions_league_matches/total_matches:.1%})")
        print(f"  Europa League: {europa_league_matches} ({europa_league_matches/total_matches:.1%})")
        print(f"\nStock return statistics:")
        print(f"  Average daily return: {mean_return:.4f} ({mean_return*100:.2f}%)")
        print(f"  Daily volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        print(f"  Positive return days: {positive_days:.1%}")
        
        print("\nAVERAGE NEXT-DAY RETURN BY MATCH OUTCOME:")
        print(f"  After BVB wins:   {win_return:+.4f} ({win_return*100:.2f}%)")
        print(f"  After BVB losses: {loss_return:+.4f} ({loss_return*100:.2f}%)")
        
        return {
            'total_matches': total_matches,
            'complete_odds_data': complete_odds_data,
            'bundesliga_matches': bundesliga_matches,
            'champions_league_matches': champions_league_matches,
            'mean_return': mean_return,
            'volatility': volatility,
            'positive_days': positive_days,
            'behavioral_validation': {
                'win_return': win_return,
                'loss_return': loss_return
            }
        }

    def analyze_alpha_signals(self, data):
        """Analyze betting odds as predictive signals for stock returns"""
        
        # Filter valid data and create helper columns
        clean_data = data.dropna(
            subset=['next_day_return', 'three_day_return', 'bvb_win_prob', 'surprise_factor', 'bookmaker_margin']
        ).copy()
        # Return over trading days 2-3, relative to the price after Day-1
        # R1 = (P1-P0)/P0, R3 = (P3-P0)/P0
        # Correction over days 2-3 should be (P3-P1)/P1 = (1+R3)/(1+R1) - 1
        clean_data['correction_return'] = (
            (1 + clean_data['three_day_return']) / (1 + clean_data['next_day_return']) - 1
        )
        
        print("\n🎯 ALPHA SIGNAL ANALYSIS")
        
        # --- Run all alpha analysis components ---
        self._analyze_correlations(clean_data)
        self._analyze_probability_ranges(clean_data)
        self._analyze_bookmaker_margins(clean_data)
        self._analyze_surprise_factor(clean_data)
        # self._analyze_strategy_examples(clean_data)  # Removed per user request
        
        # The return value can be a summary of the most important findings
        # For now, we return the main correlations
        return {
            'correlation_prob_return': clean_data['bvb_win_prob'].corr(clean_data['next_day_return']),
            'correlation_margin_vol': clean_data['bookmaker_margin'].corr(clean_data['next_day_return'].abs())
        }

    def _analyze_correlations(self, data):
        """High-level correlation analysis."""
        print(f"\n📊 CORRELATIONS")
        correlation_prob = data['bvb_win_prob'].corr(data['next_day_return'])
        correlation_margin = data['bookmaker_margin'].corr(data['next_day_return'])
        correlation_prob_vol = data['bvb_win_prob'].corr(data['next_day_return'].abs())
        correlation_margin_vol = data['bookmaker_margin'].corr(data['next_day_return'].abs())
        
        print(f"  Win Prob vs Return:    {correlation_prob:+.4f}")
        print(f"  Win Prob vs Volatility:{correlation_prob_vol:+.4f}")
        print(f"  Margin vs Return:      {correlation_margin:+.4f}")
        print(f"  Margin vs Volatility:  {correlation_margin_vol:+.4f}")

    def _analyze_probability_ranges(self, data):
        """Analyze returns based on win probability ranges."""
        print(f"\n#️⃣ PROBABILITY RANGES")
        high_prob_matches = data[data['bvb_win_prob'] > 0.6]
        low_prob_matches = data[data['bvb_win_prob'] < 0.4]
        
        if high_prob_matches.empty or low_prob_matches.empty:
            print("  Not enough data for high/low probability comparison.")
            return
            
        high_prob_return = high_prob_matches['next_day_return'].mean()
        low_prob_return = low_prob_matches['next_day_return'].mean()
        
        print(f"  High prob (>60%) return: {high_prob_return:+.4f}")
        print(f"  Low prob (<40%) return:  {low_prob_return:+.4f}")
        
        if len(high_prob_matches) > 5 and len(low_prob_matches) > 5:
            _, p_value = stats.ttest_ind(high_prob_matches['next_day_return'], low_prob_matches['next_day_return'])
            print(f"  T-test p-value: {p_value:.4f}")

    def _analyze_bookmaker_margins(self, data):
        """Analyze returns based on bookmaker margin."""
        print(f"\n💰 BOOKMAKER MARGINS")
        low_margin = data[data['bookmaker_margin'] < 0.05]
        high_margin = data[data['bookmaker_margin'] > 0.1]

        if low_margin.empty or high_margin.empty:
            print("  Not enough data for high/low margin comparison.")
            return

        low_margin_return = low_margin['next_day_return'].mean()
        high_margin_return = high_margin['next_day_return'].mean()
        low_margin_vol = low_margin['next_day_return'].abs().mean()
        high_margin_vol = high_margin['next_day_return'].abs().mean()
        
        print(f"  Low margin (<5%) : return={low_margin_return:+.4f}, vol={low_margin_vol:.4f}")
        print(f"  High margin (>10%): return={high_margin_return:+.4f}, vol={high_margin_vol:.4f}")
        
        if len(low_margin) > 5 and len(high_margin) > 5:
            _, vol_p_value = stats.ttest_ind(low_margin['next_day_return'].abs(), high_margin['next_day_return'].abs())
            print(f"  Volatility t-test p-value: {vol_p_value:.4f}")

    def _analyze_surprise_factor(self, data):
        """
        Analyzes the behavioral bias from the surprise factor by correlating
        the initial 1-day return with the subsequent 2-day correction return.
        """
        print(f"\n🎭 SURPRISE FACTOR ANALYSIS")
        
        high_surprise = data[data['surprise_factor'] > 0.7]
        n_high = len(high_surprise)
        
        if high_surprise.empty:
            print("  Not enough high-surprise data to analyze.")
            return

        # --- Behavioral Bias Test ---
        print(f"  (For events with surprise factor > 0.7) – n = {n_high}")
        
        # Primary Correlation Test:
        # A negative correlation suggests OVERREACTION (reversal).
        # A positive correlation suggests UNDERREACTION (momentum).
        if len(high_surprise) > 1:
            correction_corr = high_surprise['next_day_return'].corr(high_surprise['correction_return'])
            print(f"  Correlation(Day 1 Return vs Day 2-3 Correction): {correction_corr:.4f}")
        
        # Breakdown by wins and losses to show the pattern
        surprising_wins = high_surprise[high_surprise['bvb_won'] == 1]
        surprising_losses = high_surprise[high_surprise['bvb_won'] == 0]
        n_wins = len(surprising_wins)
        n_losses = len(surprising_losses)

        if not surprising_wins.empty:
            win_d1_return = surprising_wins['next_day_return'].mean()
            win_correction = surprising_wins['correction_return'].mean()
            print(f"\n  Surprising Wins (n = {n_wins}):")
            print(f"    Avg Day 1 Return:       {win_d1_return:+.4f}")
            print(f"    Avg Day 2-3 Correction: {win_correction:+.4f}")

        if not surprising_losses.empty:
            loss_d1_return = surprising_losses['next_day_return'].mean()
            loss_correction = surprising_losses['correction_return'].mean()
            print(f"\n  Surprising Losses (n = {n_losses}):")
            print(f"    Avg Day 1 Return:       {loss_d1_return:+.4f}")
            print(f"    Avg Day 2-3 Correction: {loss_correction:+.4f}")

