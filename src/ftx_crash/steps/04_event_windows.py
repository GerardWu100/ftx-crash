"""Notebook section: event windows."""

# Event-study windows (UTC) aligned with tz-aware spot/futures indices from step 03.
luna_estimation_start = pd.Timestamp("2022-03-01", tz="UTC")
luna_estimation_end = pd.Timestamp("2022-05-08", tz="UTC")
luna_crisis_start = pd.Timestamp("2022-05-09", tz="UTC")
luna_crisis_end = pd.Timestamp("2022-05-20", tz="UTC")
luna_post_start = pd.Timestamp("2022-05-21", tz="UTC")
luna_post_end = pd.Timestamp("2022-06-30", tz="UTC")

ftx_estimation_start = pd.Timestamp("2022-09-01", tz="UTC")
ftx_estimation_end = pd.Timestamp("2022-11-05", tz="UTC")
ftx_crisis_start = pd.Timestamp("2022-11-06", tz="UTC")
ftx_crisis_end = pd.Timestamp("2022-11-19", tz="UTC")
ftx_post_start = pd.Timestamp("2022-11-20", tz="UTC")
ftx_post_end = pd.Timestamp("2022-12-31", tz="UTC")

print("Event Windows (UTC):")
print("\nTerra/LUNA:")
print(f"  Estimation: {luna_estimation_start.date()} to {luna_estimation_end.date()}")
print(
    f"  Crisis:     {luna_crisis_start.date()} to {luna_crisis_end.date()} (UST de-peg began May 9)"
)
print(f"  Post-crisis: {luna_post_start.date()} to {luna_post_end.date()}")
print("\nFTX:")
print(f"  Estimation: {ftx_estimation_start.date()} to {ftx_estimation_end.date()}")
print(
    f"  Crisis:     {ftx_crisis_start.date()} to {ftx_crisis_end.date()} (Binance FTT sale announced Nov 6)"
)
print(f"  Post-crisis: {ftx_post_start.date()} to {ftx_post_end.date()}")

# Paper appendix table: estimation and crisis windows for both events.
timeline_data = {
    "Event": ["Terra/LUNA Collapse", "FTX Exchange Failure"],
    "Estimation Window Start": [
        luna_estimation_start.strftime("%Y-%m-%d"),
        ftx_estimation_start.strftime("%Y-%m-%d"),
    ],
    "Estimation Window End": [
        luna_estimation_end.strftime("%Y-%m-%d"),
        ftx_estimation_end.strftime("%Y-%m-%d"),
    ],
    "Crisis Window Start": [
        luna_crisis_start.strftime("%Y-%m-%d"),
        ftx_crisis_start.strftime("%Y-%m-%d"),
    ],
    "Crisis Window End": [
        luna_crisis_end.strftime("%Y-%m-%d"),
        ftx_crisis_end.strftime("%Y-%m-%d"),
    ],
    "Key Trigger Date": ["May 9, 2022 (UST De-peg)", "Nov 6, 2022 (Binance FTT Sale)"],
    "Primary Asset": ["BTC Futures (Binance)", "BTC Futures Proxy (BITO)"],
}

tab_timeline = pd.DataFrame(timeline_data).set_index("Event")
save_paper_table(tab_timeline, "tab_appendix_event_timeline")
