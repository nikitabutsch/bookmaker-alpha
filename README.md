# Betting-Market Alpha – Borussia Dortmund (BVB.DE)

> "Can the wisdom of sports-bettors predict a football club's stock – and if not, **what does** move the price?"


## 1 Snapshot
* Universe   : 971 Borussia Dortmund matches (2005-2015)  
* Data        :  
  • Closing odds from 30+ bookmakers (Kaggle *Beat-the-Bookie*)  
  • Daily OHLC for BVB.DE (Yahoo Finance)  
* Core pipeline : `feature_engineering.py` → `main.py` (analysis stats)
* Optional step : `modeling.py` – standalone CatBoost model, run separately

### Working hypotheses
1. **Odds-Momentum** – Higher implied win-probability should predict a positive next-day return (betting market leads equity).
2. **Margin-Volatility** – Larger bookmaker margin (market uncertainty) should foreshadow higher next-day volatility.
3. **Surprise Mean-Reversion** – When the actual result strongly contradicts the odds, the Day-1 price reaction will partially reverse over Days 2-3.

Hypotheses 1 & 2 were empirically rejected; Hypothesis 3 is strongly supported and forms the basis of the CatBoost model below.


## 2 The Pipeline
1. **Load Data**: odds + stock prices
2. **Feature Engineering**: win/draw/lose probs, bookmaker margin, surprise factor, match importance
3.  **Statistical Analysis**: next-day return vs. factors, mean-reversion tests
4. **Fit CatBoost Model**


## 3 Findings (printed by `main.py`)
```
🎯 ALPHA SIGNAL ANALYSIS

📊 CORRELATIONS
  Win Prob vs Return:    -0.0071
  Win Prob vs Volatility:+0.0065
  Margin vs Return:      +0.0128
  Margin vs Volatility:  +0.0115

#️⃣ PROBABILITY RANGES
  High prob (>60%) return: -0.0005
  Low prob (<40%) return:  -0.0027
  T-test p-value: 0.4085

💰 BOOKMAKER MARGINS
  Low margin (<5%) : return=-0.0013, vol=0.0234
  High margin (>10%): return=-0.0008, vol=0.0227
  Volatility t-test p-value: 0.9202

🎭 SURPRISE FACTOR ANALYSIS
  (For events with surprise factor > 0.7) – n = 370
  Correlation(Day 1 Return vs Day 2-3 Correction): -0.5435

  Surprising Wins (n = 49):
    Avg Day 1 Return:       +0.0041
    Avg Day 2-3 Correction: -0.0069

  Surprising Losses (n = 321):
    Avg Day 1 Return:       -0.0092
    Avg Day 2-3 Correction: +0.0115
```
**Translation**
* Pre-match odds do **not** lead the equity – Hypothesis #1 rejected.
* Bookmaker margin is a poor risk proxy – Hypothesis #2 rejected.
* Stock **over-reacts** to unlikely results and mean-reverts within two days – a tradeable contrarian edge.

## 4 CatBoost correction-model (high-surprise subset)
*Target definition*: we train the model to predict the **2-to-3-day stock return ("correction") that follows the initial Day-1 reaction for high-surprise matches.*

```
High-surprise sample: 370 matches
Train n = 277, Test n = 93

📋  Hold-out metrics:
  RMSE = 0.0470
  R²   = 0.320
  Sign accuracy = 66.67%

SHAP-based feature importances (percentage of total):
  next_day_return       :  67.0%
  surprise_factor       :  12.6%
  is_europa_league      :   8.9%
  bvb_away              :   4.5%
  bvb_home              :   3.0%
  is_champions_league   :   2.1%
  is_bundesliga         :   1.4%
  is_friendly           :   0.3%
  is_domestic_cup       :   0.2%
  ```

Interpretation – Day-1 move plus surprise factor allow us to forecast ~32 % of the variance in the subsequent 2-day correction and get direction right two-thirds of the time — strong evidence of predictable mean-reversion.



## 5 Blind Spots & Next Steps
* **Closing odds only** – need intraday line-moves to time entry.
* **Single issuer** – extend cross-sectionally to listed clubs or sports franchises.
* **Execution** – add liquidity/impact model; current returns are gross.
* **Walk-forward validation** – roll an expanding-window train/test to ensure out-of-time performance.
* **Cross-sectional test** – replicate on other listed clubs (MANU, JUVE, AFC.BR) to rule out sample-specific artefacts.
* **Live odds feed & paper trading** – stream real-time closing lines, generate signals and paper-trade for 1-3 months.
* **Cost-adjusted back-test** – embed bid-ask + 2 bp commission and re-evaluate Sharpe / hit-rate.
* **Risk overlay** – size positions by predicted correction magnitude; circuit-break if live hit-rate < 55 %.


## 6 Run It
```bash
pip install -r requirements.txt
python feature_engineering.py      # builds results/alpha_dataset.csv
python main.py                     # prints stats above
python modeling.py                 # CatBoost correction model metrics
```