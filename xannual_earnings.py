"""\
Read earnings_dates_all.csv (produced by a yfinance earnings script) and summarize by symbol and calendar year.

For each (symbol, year) combination, output:
  - n_earnings_dates: number of distinct earnings_date values in that year
  - first_earnings_date, last_earnings_date (within that year)
  - reported_eps_q1..reported_eps_q4: quarterly actual EPS (based on announcement date)
  - eps_estimate_q1..eps_estimate_q4: quarterly estimated EPS (based on announcement date)

Rules:
  - Quarter assignment is based on earnings_date (the announcement date).
  - If more than one earnings report appears in the same quarter, use the first report
    (earliest earnings_datetime_utc; if missing, earliest earnings_date).
  - If a quarter has no earnings report, show NaN for that quarter.

Outputs are ASCII and written to annual_earnings.csv.
"""

import numpy as np
import pandas as pd


# -----------------------
# User settings
# -----------------------
input_csv = "earnings_dates_all.csv"
output_csv = "annual_earnings.csv"

# If True, also print the output CSV to stdout
print_to_stdout = False


# -----------------------
# Helpers
# -----------------------
def _to_float_series(x: pd.Series) -> pd.Series:
    """Convert strings like '+9.77', '-39.16', '-', '' to float (NaN on missing)."""
    s = x.astype("string")
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace("+", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def _date_to_iso(x) -> object:
    if pd.isna(x):
        return pd.NA
    # x may be Timestamp or datetime.date
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return pd.NA


def main() -> None:
    df = pd.read_csv(input_csv)

    # normalize column names
    if "ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"ticker": "symbol"})

    if "symbol" not in df.columns:
        raise ValueError("Missing required column: symbol (or ticker)")

    # parse dates
    df["earnings_date"] = pd.to_datetime(df.get("earnings_date"), errors="coerce")
    df["earnings_datetime_utc"] = pd.to_datetime(df.get("earnings_datetime_utc"), errors="coerce", utc=True)

    # numeric fields
    if "EPS Estimate" in df.columns:
        df["eps_estimate"] = _to_float_series(df["EPS Estimate"])
    else:
        df["eps_estimate"] = np.nan

    if "Reported EPS" in df.columns:
        df["reported_eps"] = _to_float_series(df["Reported EPS"])
    else:
        df["reported_eps"] = np.nan

    # keep only rows with a valid announcement date
    df = df.loc[df["earnings_date"].notna()].copy()
    if df.empty:
        out = pd.DataFrame(
            columns=[
                "symbol",
                "year",
                "n_earnings_dates",
                "first_earnings_date",
                "last_earnings_date",
                "reported_eps_q1",
                "reported_eps_q2",
                "reported_eps_q3",
                "reported_eps_q4",
                "eps_estimate_q1",
                "eps_estimate_q2",
                "eps_estimate_q3",
                "eps_estimate_q4",
            ]
        )
        out.to_csv(output_csv, index=False, na_rep="nan")
        print(f"Wrote {output_csv} (0 rows)")
        return

    df["year"] = df["earnings_date"].dt.year.astype(int)
    df["quarter"] = df["earnings_date"].dt.quarter.astype(int)

    # Build a sortable timestamp: use earnings_datetime_utc if present, else earnings_date at midnight UTC.
    fallback_dt = pd.to_datetime(df["earnings_date"], errors="coerce").dt.tz_localize("UTC")
    df["sort_dt"] = df["earnings_datetime_utc"].where(df["earnings_datetime_utc"].notna(), fallback_dt)

    # Choose the first report per (symbol, year, quarter)
    df_q = (
        df.sort_values(["symbol", "year", "quarter", "sort_dt"])
        .groupby(["symbol", "year", "quarter"], sort=False)
        .head(1)
        .copy()
    )

    # Year-level stats per (symbol, year)
    df_y = df.groupby(["symbol", "year"], as_index=False).agg(
        n_earnings_dates=("earnings_date", lambda s: int(pd.Series(s).nunique(dropna=True))),
        first_earnings_date=("earnings_date", "min"),
        last_earnings_date=("earnings_date", "max"),
    )
    df_y["first_earnings_date"] = df_y["first_earnings_date"].map(_date_to_iso)
    df_y["last_earnings_date"] = df_y["last_earnings_date"].map(_date_to_iso)

    # Pivot quarterly actuals and estimates
    rep_wide = df_q.pivot(index=["symbol", "year"], columns="quarter", values="reported_eps")
    est_wide = df_q.pivot(index=["symbol", "year"], columns="quarter", values="eps_estimate")

    # Ensure columns 1..4 exist
    for q in [1, 2, 3, 4]:
        if q not in rep_wide.columns:
            rep_wide[q] = np.nan
        if q not in est_wide.columns:
            est_wide[q] = np.nan

    rep_wide = rep_wide[[1, 2, 3, 4]].rename(
        columns={1: "reported_eps_q1", 2: "reported_eps_q2", 3: "reported_eps_q3", 4: "reported_eps_q4"}
    )
    est_wide = est_wide[[1, 2, 3, 4]].rename(
        columns={1: "eps_estimate_q1", 2: "eps_estimate_q2", 3: "eps_estimate_q3", 4: "eps_estimate_q4"}
    )

    out = df_y.merge(rep_wide.reset_index(), on=["symbol", "year"], how="left")
    out = out.merge(est_wide.reset_index(), on=["symbol", "year"], how="left")

    # Final column order
    cols = [
        "symbol",
        "year",
        "n_earnings_dates",
        "first_earnings_date",
        "last_earnings_date",
        "reported_eps_q1",
        "reported_eps_q2",
        "reported_eps_q3",
        "reported_eps_q4",
        "eps_estimate_q1",
        "eps_estimate_q2",
        "eps_estimate_q3",
        "eps_estimate_q4",
    ]
    out = out[cols].sort_values(["symbol", "year"]).reset_index(drop=True)

    out.to_csv(output_csv, index=False, na_rep="nan")
    print(f"Wrote {output_csv} with {len(out)} rows")

    if print_to_stdout:
        # Print as CSV for easy piping.
        print(out.to_csv(index=False, lineterminator="\n", na_rep="nan"))

    # Print distribution of n_earnings_dates across (symbol, year) combinations.
    vc = out["n_earnings_dates"].value_counts(dropna=False)
    max_n = int(out["n_earnings_dates"].max()) if len(out) else 0

    print("n_earnings_dates  n_symbol_year")
    print("-----------------------------")
    for n in range(1, max_n + 1):
        cnt = int(vc.get(n, 0))
        print(f"{n:15d}  {cnt:12d}")


if __name__ == "__main__":
    main()
