# Betting-Market Alpha – Borussia Dortmund (BVB.DE)

> "Can the wisdom of sports-bettors predict a football club's stock – and if not, **what does** move the price?"


## 1 Snapshot
* Universe   : 971 Borussia Dortmund matches (2005-2015)  
* Data        :  
  • Closing odds from 30+ bookmakers (Kaggle *Beat-the-Bookie*)  
  • Daily OHLC for BVB.DE (Yahoo Finance)  
* Code path   : `feature_engineering.py` → `main.py` → **console output only** – no notebooks, no dashboards.


## 2 What Happens Under the Hood
```
1  Load & tidy        – odds  + stock prices
2  Engineer factors   – win/draw/lose probs, bookmaker margin, SURPRISE factor
3  Analyse            – next-day return vs. factors, mean-reversion tests
```


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

### CatBoost correction-model (high-surprise subset)
*Target definition*: we train the model to predict the **2-to-3-day stock return ("correction") that follows the initial Day-1 reaction for high-surprise matches.*

```
📋  Hold-out metrics:
  RMSE = 0.0470
  R²   = 0.320
  Sign accuracy = 66.67%

Feature importances:
  next_day_return       :  62.9
  is_europa_league      :  15.0
  surprise_factor       :  14.4
  bvb_away              :   2.7
  bvb_home              :   2.0
  is_champions_league   :   1.3
  is_bundesliga         :   1.2
  is_friendly           :   0.4
  is_domestic_cup       :   0.1
```

Interpretation – Day-1 move plus surprise factor allow us to forecast ~32 % of the variance in the subsequent 2-day correction and get direction right two-thirds of the time — strong evidence of predictable mean-reversion.



## 4 Blind Spots & Next Steps
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