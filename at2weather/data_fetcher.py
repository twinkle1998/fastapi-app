import requests
import pandas as pd
import time
from pathlib import Path
from datetime import date

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
SYDNEY_LATITUDE = -33.8678
SYDNEY_LONGITUDE = 151.2073
CUTOFF_DATE = date(2025, 9, 25)

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "sunrise", "sunset", "daylight_duration", "sunshine_duration",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration"
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def fetch_weather_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weather data for a given date range."""
    params = {
        "latitude": SYDNEY_LATITUDE,
        "longitude": SYDNEY_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Australia/Sydney"
    }

    for attempt in range(3):  # retry up to 3 times
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "daily" not in data:
                print(f" No 'daily' key in API response for {start_date} → {end_date}")
                return pd.DataFrame()
            df = pd.DataFrame(data["daily"])
            if "time" not in df.columns:
                raise KeyError("❌ No 'time' column in API response")
            return df
        except requests.exceptions.RequestException as e:
            print(f"API error ({start_date} → {end_date}): {e}")
            time.sleep(5 * (attempt + 1))  # exponential backoff

    return pd.DataFrame()


def fetch_and_save_sydney_data(
    start_year: int = 2015,
    end_year: int = 2025,
    output_filename: str = "sydney_weather_full_2015_2025.csv"
) -> pd.DataFrame:
    """Fetch Sydney weather data in chunks and save."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / output_filename

    # If cached, load it instead of hitting API again
    if output_path.exists():
        print(f"Using cached dataset: {output_path}")
        return pd.read_csv(output_path, parse_dates=["time"])

    dfs = []
    for year in range(start_year, end_year + 1, 2):  # chunks of 2 years
        start_date = f"{year}-01-01"
        if year + 1 < end_year:
            end_date = f"{year+1}-12-31"
        else:
            end_date = CUTOFF_DATE.strftime("%Y-%m-%d")

        print(f"Fetching {start_date} → {end_date}")
        df_chunk = fetch_weather_range(start_date, end_date)
        if not df_chunk.empty:
            dfs.append(df_chunk)
        time.sleep(2)  # avoid 429 by spacing calls

    if not dfs:
        print("❌ No data fetched at all")
        return pd.DataFrame()

    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")
    print("Shape:", df_raw.shape)
    print("Date range:", df_raw['time'].min(), "→", df_raw['time'].max())
    return df_raw


if __name__ == "__main__":
    df = fetch_and_save_sydney_data()
    print(df.head())
