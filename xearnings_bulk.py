"""
Download historical earnings announcement datetimes from Yahoo Finance via yfinance.

- For each ticker, repeatedly calls yfinance.Ticker.get_earnings_dates(limit, offset)
  to page backward in time until reaching start_date.
- Outputs a single CSV (and optionally one CSV per ticker).

Robustness changes:
- Each ticker is fetched in a separate process with a hard wall-clock timeout, so the
  program never hangs on any single symbol.
- Missing values in any column are kept as NaN/NA (or left as-is) and do not stop the run.
"""

import time
t_start = time.perf_counter()
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf
import multiprocessing as mp


# -----------------------
# User settings
# -----------------------
max_tickers = None
ticker_file = "itot_tickers_20260115.txt"
start_date: str = "2002-01-01"         # YYYY-MM-DD
chunk_size: int = 100                  # rows per page
sleep_seconds: float = 0.25            # be polite to Yahoo
max_pages: Optional[int] = None        # set to an int to cap paging (e.g. 40)

# hard stop per ticker (prevents hanging forever)
max_seconds_per_ticker: float = 60.0

write_one_csv_per_ticker: bool = False
output_csv_all: str = "earnings_dates_all.csv"
output_dir_per_ticker: str = "."      # used only if write_one_csv_per_ticker=True


# -----------------------
# Helpers
# -----------------------
def load_tickers() -> List[str]:
    if ticker_file is None:
        tickers: List[str] = ["AAPL", "NVDA"]  # ["AAPL", "MSFT", "AMZN"]
    else:
        tickers = open(ticker_file, "r").readlines()
        tickers = [x.strip() for x in tickers if x.strip()]
    return tickers


def _normalize_earnings_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy().reset_index()

    if "Earnings Date" in tmp.columns:
        dt_col = "Earnings Date"
    else:
        dt_col = tmp.columns[0]
        tmp = tmp.rename(columns={dt_col: "Earnings Date"})
        dt_col = "Earnings Date"

    tmp["ticker"] = ticker

    dt = pd.to_datetime(tmp[dt_col], errors="coerce", utc=True)
    tmp["earnings_datetime_utc"] = dt
    tmp["earnings_date"] = tmp["earnings_datetime_utc"].dt.date

    # Keep going even if some columns/rows are missing. Normalize common "missing" markers.
    for c in ["EPS Estimate", "Reported EPS", "Surprise(%)", "Surprise (%)"]:
        if c in tmp.columns:
            tmp[c] = tmp[c].replace("-", pd.NA)

    # Some feeds use "Surprise (%)" instead of "Surprise(%)"
    if "Surprise (%)" in tmp.columns and "Surprise(%)" not in tmp.columns:
        tmp = tmp.rename(columns={"Surprise (%)": "Surprise(%)"})

    return tmp


def fetch_earnings_dates_back_to(
    ticker: str,
    start_dt_utc: pd.Timestamp,
    chunk: int = 50,
    sleep_s: float = 0.25,
    max_pages: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    offset = 0
    pages = 0
    tkr = yf.Ticker(ticker)

    last_min_dt = None

    while True:
        try:
            df = tkr.get_earnings_dates(limit=chunk, offset=offset)
        except Exception as e:
            print(f"{ticker}: error calling get_earnings_dates(limit={chunk}, offset={offset}): {e}")
            break

        if df is None or df.empty:
            break

        norm = _normalize_earnings_df(df, ticker)
        if norm.empty:
            break

        rows.append(norm)

        pages += 1
        if max_pages is not None and pages >= max_pages:
            break

        min_dt = norm["earnings_datetime_utc"].min()

        # Stagnation guard (avoid infinite paging if provider repeats the same block)
        if last_min_dt is not None and pd.notna(min_dt) and pd.notna(last_min_dt) and min_dt >= last_min_dt:
            # print(f"{ticker}: paging not moving earlier than {last_min_dt}; stopping.")
            break
        last_min_dt = min_dt

        if pd.notna(min_dt) and min_dt <= start_dt_utc:
            break

        offset += chunk
        time.sleep(sleep_s)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    out = out.drop_duplicates(subset=["ticker", "earnings_datetime_utc"])
    out = out[out["earnings_datetime_utc"] >= start_dt_utc].copy()
    out = out.sort_values(["ticker", "earnings_datetime_utc"]).reset_index(drop=True)

    return out


def _worker_fetch(
    ticker: str,
    start_date_str: str,
    chunk: int,
    sleep_s: float,
    max_pages: Optional[int],
    q: mp.Queue,
) -> None:
    try:
        start_dt_utc = pd.Timestamp(start_date_str)
        if start_dt_utc.tzinfo is None:
            start_dt_utc = start_dt_utc.tz_localize("UTC")
        else:
            start_dt_utc = start_dt_utc.tz_convert("UTC")

        df = fetch_earnings_dates_back_to(
            ticker=ticker,
            start_dt_utc=start_dt_utc,
            chunk=chunk,
            sleep_s=sleep_s,
            max_pages=max_pages,
        )
        q.put(("ok", ticker, df))
    except Exception as e:
        q.put(("err", ticker, str(e)))


def fetch_one_ticker_with_timeout(
    ticker: str,
    start_date_str: str,
    chunk: int,
    sleep_s: float,
    max_pages: Optional[int],
    timeout_s: float,
) -> Tuple[str, str, object]:
    """
    Returns ("ok", ticker, df) or ("timeout", ticker, None) or ("err", ticker, errmsg).
    Hard timeout enforced by terminating the process.
    """
    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=_worker_fetch,
        args=(ticker, start_date_str, chunk, sleep_s, max_pages, q),
        daemon=True,
    )
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(2.0)
        return ("timeout", ticker, None)

    if not q.empty():
        return q.get()

    return ("err", ticker, "no result returned from worker")


def main() -> None:
    tickers = load_tickers()
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    print("start_date:", start_date)
    print("#tickers:", len(tickers))

    all_frames = []
    ntickers = len(tickers)
    for ii, ticker in enumerate(tickers):
        print(str(ii+1) + "/" + str(ntickers), "%6s"%ticker, end="")
        status, sym, payload = fetch_one_ticker_with_timeout(
            ticker=ticker,
            start_date_str=start_date,
            chunk=chunk_size,
            sleep_s=sleep_seconds,
            max_pages=max_pages,
            timeout_s=max_seconds_per_ticker,
        )

        if status == "timeout":
            print(f"  {sym}: timed out after {max_seconds_per_ticker} seconds; skipping.")
            continue

        if status == "err":
            print(f"  {sym}: error: {payload}")
            continue

        df_t: pd.DataFrame = payload
        if df_t is None or df_t.empty:
            print(f"  {sym}: no data returned.")
            continue

        nrows = len(df_t)
        d0 = df_t["earnings_datetime_utc"].min()
        d1 = df_t["earnings_datetime_utc"].max()

        s0 = d0.date().isoformat() if pd.notna(d0) else "na"
        s1 = d1.date().isoformat() if pd.notna(d1) else "na"

        print(f" {nrows} dates {s0} to {s1}")
        all_frames.append(df_t)

        if write_one_csv_per_ticker:
            out_path = f"{output_dir_per_ticker.rstrip('/')}/{sym}_earnings_dates.csv"
            df_t.to_csv(out_path, index=False)
            print(f"  wrote {out_path}")

    if not all_frames:
        print("No earnings data returned for any ticker.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)

    desired = [
        "Earnings Date",
        "earnings_datetime_utc",
        "earnings_date",
        "ticker",
        "EPS Estimate",
        "Reported EPS",
        "Surprise(%)",
    ]
    cols = [c for c in desired if c in df_all.columns] + [c for c in df_all.columns if c not in desired]
    df_all = df_all.loc[:, cols]

    df_all.to_csv(output_csv_all, index=False)
    print(f"\nWrote {output_csv_all} with {len(tickers)} symbols, {len(df_all)} total rows.")


if __name__ == "__main__":
    try:
        mp.freeze_support()
        main()
    finally:
        elapsed = time.perf_counter() - t_start
        print(f"time elapsed: {elapsed:.3f} seconds")
