# Methodology

Written for a reviewer or auditor: what is computed, from what, and with which conventions.

## Inputs
- **Risk factors** (daily, FRED): `config.SERIES`. Price factors use simple returns `P_t/P_{t-1} - 1`;
  rate/spread factors use absolute changes in percentage points divided by 100 (computed in the SQL view
  `factor_changes` with `LAG() OVER (PARTITION BY series_id ORDER BY obs_date)`).
- **Calendar:** the S&P 500's observation dates. Other series are forward-filled up to 5 consecutive
  missing days; a date with a remaining gap in any series is dropped for all series. Start = latest first-valid
  date across series.
- **Positions:** synthetic, seeded (see `portfolio.py`). `notional_usd` is signed market value.

## Valuation
- Delta-one: `P&L = N * r`. For JPY (quoted JPY per USD) the USD value of the holding moves by `1/(1+r) - 1`.
- Fixed-rate bonds: flat-yield price per 100, periodic compounding, `P(y) = c/f * (1-(1+y/f)^-n)/(y/f) + 100(1+y/f)^-n`,
  `n = maturity*freq`. P&L for a scenario = `N * (P(y0 + dy)/P(y0) - 1)`. `y0` is today's benchmark yield
  (+ OAS for credit bonds); `dy` is the historical move in the benchmark (+ the OAS move).

## Measures (1-day horizon, lookback window = last 500 scenarios by default)
- **Historical VaR** = `-quantile(P&L, 1 - cl)` (linear interpolation). **ES** = mean loss over scenarios at or beyond that quantile.
- **Parametric VaR** = `z_cl * stdev(P&L)` (zero mean, sample stdev).
- **DV01** = `N * (P(y-1bp) - P(y+1bp)) / (2*P(y))` per bond, USD per bp (gain when yields fall).
  **CS01** = loss on a +1bp widening of the OAS for credit bonds (`-DV01` of those bonds, as spread enters yield 1:1).
- **Standalone VaR by asset class** and **diversification benefit** = sum of standalone VaRs - portfolio VaR.
- **Backtest:** for each of the last `min(250, N - lookback)` days, VaR is estimated from the preceding `lookback`
  hypothetical P&Ls of the constant book; an exception is `P&L_t < -VaR_t`. Kupiec POF likelihood-ratio test, chi-square(1).

## Data-quality checks (`dq_results`)
Row reconciliation (source dates vs rows loaded), duplicate dates, missing markers, non-positive prices,
robust-sigma outliers (kept, flagged), forward-fill usage, staleness vs panel end, panel date count, history depth.
`warn` = investigate; `fail` = the load is not trustworthy.

## Known limits
See "Honest scope" in the README.
