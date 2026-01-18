"""
Read an earnings_dates_all.csv produced by the yfinance script and compute per-symbol summary stats.

Output columns (all ascii):
  symbol
  n_earnings_dates
  first_earnings_date
  last_earnings_date              (last date with Reported EPS present and <= now)
  next_earnings_date              (next scheduled date with earnings_datetime_utc > now)

  median_surprise
  mean_surprise
  std_surprise
  n_positive_surprises
  n_negative_surprises
  n_zero_surprises
  frac_positive_surprises
  frac_negative_surprises
  frac_zero_surprises

  year_first_full_earnings
  year_last_full_earnings
  eps_first_full_year
  eps_last_full_year
  eps_cagr                          (computed only if eps endpoints > 0 and years differ)

  median_days_between_earning_dates (based on reported earnings dates only)

  min_days_between_earning_dates    (based on reported earnings dates only)
  max_days_between_earning_dates    (based on reported earnings dates only)

"Full year earnings" = sum of quarterly Reported EPS within a calendar year, requiring 4 non-missing quarters.
"""

import numpy as np
import pandas as pd


# -----------------------
# User settings
# -----------------------
input_csv = "earnings_dates_all.csv"
output_csv = "earnings_summary.csv"


# -----------------------
# Helpers
# -----------------------
def _to_float_series(x: pd.Series) -> pd.Series:
    # Accept numbers or strings like "+9.77", "-39.16", "-", "".
    s = x.astype("string")
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace("+", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def _first_non_na(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return pd.NA
    return s2.iloc[0]


def summarize_one_symbol(df_sym: pd.DataFrame, now_utc: pd.Timestamp) -> dict:
    sym = df_sym["symbol"].iloc[0]

    # count and first date (use all rows, including those with missing EPS)
    n_dates = df_sym["earnings_date"].nunique(dropna=True)
    first_date = df_sym["earnings_date"].min()

    # last reported earnings date (Reported EPS present, and in the past)
    mask_reported = df_sym["reported_eps"].notna() & (df_sym["earnings_datetime_utc"] <= now_utc)
    last_reported_date = df_sym.loc[mask_reported, "earnings_date"].max()

    # next scheduled earnings date (strictly in the future)
    mask_future = df_sym["earnings_datetime_utc"] > now_utc
    next_date = df_sym.loc[mask_future, "earnings_date"].min()

    # surprise stats (use past surprises only)
    mask_surprise = df_sym["surprise_pct"].notna() & (df_sym["earnings_datetime_utc"] <= now_utc)
    s = pd.to_numeric(df_sym.loc[mask_surprise, "surprise_pct"], errors="coerce")
    s = s.dropna()

    if len(s) == 0:
        median_surprise = np.nan
        mean_surprise = np.nan
        std_surprise = np.nan
        n_pos = 0
        n_neg = 0
        n_zero = 0
        frac_pos = np.nan
        frac_neg = np.nan
        frac_zero = np.nan
    else:
        median_surprise = float(s.median())
        mean_surprise = float(s.mean())
        std_surprise = float(s.std(ddof=1)) if len(s) >= 2 else np.nan

        eps0 = 1e-12
        n_pos = int((s > eps0).sum())
        n_neg = int((s < -eps0).sum())
        n_zero = int((s.abs() <= eps0).sum())

        denom = float(len(s))
        frac_pos = n_pos / denom
        frac_neg = n_neg / denom
        frac_zero = n_zero / denom

    # quarterly set for "full year" EPS and max gap:
    # - use only reported EPS
    # - use only past rows
    df_rep = df_sym.loc[mask_reported, ["earnings_date", "earnings_datetime_utc", "reported_eps"]].copy()

    # min/median/max days between earning dates (reported only)
    if len(df_rep) >= 2:
        dts = df_rep.sort_values("earnings_date")["earnings_date"]
        gaps = dts.diff().dt.days.dropna()
        if len(gaps) == 0:
            min_gap_days = np.nan
            median_gap_days = np.nan
            max_gap_days = np.nan
        else:
            min_gap = gaps.min()
            median_gap = gaps.median()
            max_gap = gaps.max()
            min_gap_days = int(min_gap) if pd.notna(min_gap) else np.nan
            median_gap_days = float(median_gap) if pd.notna(median_gap) else np.nan
            max_gap_days = int(max_gap) if pd.notna(max_gap) else np.nan
    else:
        min_gap_days = np.nan
        median_gap_days = np.nan
        max_gap_days = np.nan

    # full-year EPS: calendar year sum of quarterly reported EPS, require 4 quarters
    if len(df_rep) == 0:
        year_first_full = np.nan
        year_last_full = np.nan
        eps_first_full = np.nan
        eps_last_full = np.nan
        eps_cagr = np.nan
    else:
        # dedupe by earnings_date; keep first non-missing reported_eps
        df_rep2 = df_rep.sort_values("earnings_datetime_utc").groupby("earnings_date", as_index=False).agg(
            reported_eps=("reported_eps", _first_non_na)
        )
        df_rep2["reported_eps"] = pd.to_numeric(df_rep2["reported_eps"], errors="coerce")
        df_rep2["year"] = df_rep2["earnings_date"].dt.year

        ann = df_rep2.groupby("year", as_index=False).agg(
            eps_sum=("reported_eps", "sum"),
            n_quarters=("reported_eps", "count"),
        )
        ann_full = ann.loc[ann["n_quarters"] == 4].copy()

        if ann_full.empty:
            year_first_full = np.nan
            year_last_full = np.nan
            eps_first_full = np.nan
            eps_last_full = np.nan
            eps_cagr = np.nan
        else:
            year_first_full = int(ann_full["year"].min())
            year_last_full = int(ann_full["year"].max())

            eps_first_full = float(ann_full.loc[ann_full["year"] == year_first_full, "eps_sum"].iloc[0])
            eps_last_full = float(ann_full.loc[ann_full["year"] == year_last_full, "eps_sum"].iloc[0])

            n_years = year_last_full - year_first_full
            if n_years > 0 and eps_first_full > 0.0 and eps_last_full > 0.0:
                eps_cagr = float((eps_last_full / eps_first_full) ** (1.0 / n_years) - 1.0)
            else:
                eps_cagr = np.nan

    def _date_to_iso(x):
        if pd.isna(x):
            return pd.NA
        return x.date().isoformat()

    return {
        "symbol": sym,
        "n_earnings_dates": int(n_dates) if pd.notna(n_dates) else 0,
        "first_earnings_date": _date_to_iso(first_date),
        "last_earnings_date": _date_to_iso(last_reported_date),
        "next_earnings_date": _date_to_iso(next_date),

        "median_surprise": median_surprise,
        "mean_surprise": mean_surprise,
        "std_surprise": std_surprise,

        "n_positive_surprises": n_pos,
        "n_negative_surprises": n_neg,
        "n_zero_surprises": n_zero,

        "frac_positive_surprises": frac_pos,
        "frac_negative_surprises": frac_neg,
        "frac_zero_surprises": frac_zero,

        "year_first_full_earnings": year_first_full,
        "year_last_full_earnings": year_last_full,
        "eps_first_full_year": eps_first_full,
        "eps_last_full_year": eps_last_full,
        "eps_cagr": eps_cagr,

        "median_days_between_earning_dates": median_gap_days,

        "min_days_between_earning_dates": min_gap_days,

        "max_days_between_earning_dates": max_gap_days,
    }


def main() -> None:
    df = pd.read_csv(input_csv)

    # normalize column names
    if "ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    if "Ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"Ticker": "symbol"})

    # earnings datetime and date
    if "earnings_datetime_utc" in df.columns:
        df["earnings_datetime_utc"] = pd.to_datetime(df["earnings_datetime_utc"], errors="coerce", utc=True)
    else:
        # fallback: try parsing an "Earnings Date" column as UTC
        if "Earnings Date" in df.columns:
            df["earnings_datetime_utc"] = pd.to_datetime(df["Earnings Date"], errors="coerce", utc=True)
        else:
            df["earnings_datetime_utc"] = pd.NaT

    if "earnings_date" in df.columns:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    elif "Earnings Date" in df.columns:
        df["earnings_date"] = pd.to_datetime(df["Earnings Date"], errors="coerce")
    else:
        df["earnings_date"] = pd.NaT

    # numeric fields (handle missing markers)
    if "EPS Estimate" in df.columns:
        df["eps_estimate"] = _to_float_series(df["EPS Estimate"])
    else:
        df["eps_estimate"] = pd.Series([pd.NA] * len(df))

    if "Reported EPS" in df.columns:
        df["reported_eps"] = _to_float_series(df["Reported EPS"])
    else:
        df["reported_eps"] = pd.Series([pd.NA] * len(df))

    # Surprise column: handle a few common header variants
    surprise_candidates = [
        "Surprise(%)",
        "Surprise (%)",
        "Surprise%",
        "Surprise %",
        "surprise_pct",
    ]
    surprise_col = None
    for c in surprise_candidates:
        if c in df.columns:
            surprise_col = c
            break

    if surprise_col is None:
        df["surprise_pct"] = pd.Series([pd.NA] * len(df))
    else:
        df["surprise_pct"] = _to_float_series(df[surprise_col])

    # basic sanity
    if "symbol" not in df.columns:
        raise ValueError("missing symbol column (expected 'ticker' or 'symbol')")

    now_utc = pd.Timestamp.now(tz="UTC")

    out_rows = []
    for _, df_sym in df.sort_values(["symbol", "earnings_datetime_utc"]).groupby("symbol", sort=True):
        out_rows.append(summarize_one_symbol(df_sym, now_utc))

    out = pd.DataFrame(out_rows)

    # enforce column order
    col_order = [
        "symbol",
        "n_earnings_dates",
        "first_earnings_date",
        "last_earnings_date",
        "next_earnings_date",

        "median_surprise",
        "mean_surprise",
        "std_surprise",
        "n_positive_surprises",
        "n_negative_surprises",
        "n_zero_surprises",
        "frac_positive_surprises",
        "frac_negative_surprises",
        "frac_zero_surprises",

        "year_first_full_earnings",
        "year_last_full_earnings",
        "eps_first_full_year",
        "eps_last_full_year",
        "eps_cagr",

        "median_days_between_earning_dates",
        "min_days_between_earning_dates",
        "max_days_between_earning_dates",
    ]

    out = out.reindex(columns=col_order)
    out.to_csv(output_csv, index=False)
    print(f"wrote {len(out)} symbols to {output_csv}")


if __name__ == "__main__":
    main()
