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


## 4 Blind Spots & Obvious Next Steps
* **Closing odds only** – need intraday line-moves to time entry.
* **Single issuer** – extend cross-sectionally to listed clubs or sports franchises.
* **Execution** – add liquidity/impact model; current returns are gross.


## 6 Run It
```bash
pip install -r requirements.txt
python feature_engineering.py      # builds results/alpha_dataset.csv
python main.py                     # prints stats above
```
*(Requires `closing_odds*.csv.gz` in `data/`; drop files manually or configure Kaggle token to auto-download.)* 