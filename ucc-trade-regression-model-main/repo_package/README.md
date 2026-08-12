# ucc-trade-regression-model

Regression backing our UCC trade: Long Northern Virginia Winter Energy-
Scarcity Convexity (long options on Transco Z5 basis + PJM Dominion spark
spread).

Testing: Does a cold day hit Virginia's power/gas system
harder now than it used to, since data centers ate the reserve cushion?
If yes, that's why our options are cheap relative to what they'd pay out.

## scripts

**`reserve_margin_regression.py`** — done, real data. Regresses PJM
Dominion load on HDD (cold days) + a recent-winters flag. Result: cold
sensitivity is up 24% in the last 3 winters vs before (p < 0.0001, R² =
0.73). Proves the demand side.

**`z5_convexity_model.py`** — code's ready, just needs real data. Same
regression but on Z5 basis instead of load, since that's the actual leg
we're trading. Running on fake placeholder data right now (clearly
labeled, don't use those numbers). Drop a real CSV into `data/` and it'll
auto-switch to real data.

## data

- `iad_all_hdd.csv` — daily cold weather data for Dulles, 2015–now (from IEM)
- `dominion_daily_load.csv` — daily PJM Dominion load, 2016–now (from PJM Data Miner)
- `load_price_merged_winter_data.csv` — the two merged together, winter months only
- `transco_z5_basis.csv` — doesn't exist yet, need to pull from Bloomberg

## how to run

```bash
pip install pandas numpy scipy matplotlib
python3 reserve_margin_regression.py
python3 z5_convexity_model.py
```

run from inside the folder or the file paths won't work.

## still need to do

- [ ] pull Z5 basis history off Bloomberg, save as `data/transco_z5_basis.csv`
- [ ] rerun `z5_convexity_model.py` for the real numbers + chart
- [ ] double check the "recent era = last 3 winters" cutoff is the one we want
- [ ] drop the final chart into the one-pager
