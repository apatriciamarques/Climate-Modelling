import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional

class LithuanianWeatherAPI:
    def __init__(self):
        self.base_url = "https://api.meteo.lt/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherDataCollector/1.0 (Python Script)',
            'Accept': 'application/json'
        })
        self.min_delay = 0.4

    def get_stations(self) -> List[Dict]:
        print("List of station")
        try:
            response = self.session.get(f"{self.base_url}/stations")
            response.raise_for_status()
            stations = response.json()
            print(f"Num of all station{len(stations)}")
            return stations
        except Exception as e:
            print(f"Error: {e}")
            return []

    def get_station_data_range(self, station_code: str) -> Optional[Dict]:
        try:
            time.sleep(self.min_delay)
            response = self.session.get(f"{self.base_url}/stations/{station_code}/observations")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error for: {station_code}: {e}")
            return None

    def get_observations_for_day(self, station_code: str, date: str) -> Optional[Dict]:
        """Get data from specified station code and date"""
        try:
            time.sleep(self.min_delay)
            response = self.session.get(f"{self.base_url}/stations/{station_code}/observations/{date}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            else:
                print(f" HTTP error for {station_code} by {date}: {e}")
                return None
        except Exception as e:
            print(f"Error for {station_code} by {date}: {e}")
            return None

    def download_station_data(self, station_code: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        all_observations = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        day_count = 0

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            day_count += 1
            daily_data = self.get_observations_for_day(station_code, date_str)

            if daily_data and 'observations' in daily_data:
                for obs in daily_data['observations']:
                    obs['station_code'] = daily_data['station']['code']
                    obs['station_name'] = daily_data['station']['name']
                    obs['coordinates'] = daily_data['station'].get('coordinates')
                    obs['date'] = date_str
                all_observations.extend(daily_data['observations'])

            current_date += timedelta(days=1)

        print(f"\rDownload  {len(all_observations)} records for {station_code} by {total_days} days.  ")
        return all_observations

    def download_all_stations_data(self, stations_to_process: List[Dict], start_date: datetime,
                                   end_date: datetime) -> pd.DataFrame:
        all_data = []

        effective_end_date = min(end_date, datetime.now() - timedelta(days=1))

        for i, station in enumerate(stations_to_process, 1):
            station_code = station['code']

            data_range = self.get_station_data_range(station_code)
            if data_range and 'observationsDataRange' in data_range and data_range['observationsDataRange']:
                available_start = datetime.fromisoformat(
                    data_range['observationsDataRange']['startTimeUtc'].replace('Z', '+00:00')).replace(tzinfo=None)
                available_end = datetime.fromisoformat(
                    data_range['observationsDataRange']['endTimeUtc'].replace('Z', '+00:00')).replace(tzinfo=None)

                print(
                    f" Data accessed from: {available_start.strftime('%Y-%m-%d')} up to {available_end.strftime('%Y-%m-%d')}")

                actual_start = max(start_date, available_start)
                actual_end = min(effective_end_date, available_end)

                if actual_start <= actual_end:
                    station_data = self.download_station_data(station_code, actual_start, actual_end)
                    all_data.extend(station_data)
                else:
                    print(
                        f"  No data at specificated date ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}).")
            else:
                print(f" Error")

            time.sleep(1)

        if all_data:
            df = pd.DataFrame(all_data)
            return df
        else:
            return pd.DataFrame()


def main():
    print("Download data from official LT API")
    print("=" * 70)

    # --- ВКАЖІТЬ КОДИ ПОТРІБНИХ СТАНЦІЙ ТУТ ---
    TARGET_STATIONS = [
        "kauno-ams",
        "vilniaus-ams",
        "klaipedos-ams"
    ]
    # -----------------------------------------

    api = LithuanianWeatherAPI()

    # 1. Отримати повний список станцій
    all_stations = api.get_stations()
    if not all_stations:
        print("Error. Empty list of station.")
        return

    # 2. Відфільтрувати станції за вашим списком
    stations_to_process = [st for st in all_stations if st['code'] in TARGET_STATIONS]

    found_codes = {st['code'] for st in stations_to_process}
    missing_codes = set(TARGET_STATIONS) - found_codes

    print("-" * 70)
    print(f"Found  {len(stations_to_process)}  from {len(TARGET_STATIONS)} station")
    if missing_codes:
        print(f"No station: {', '.join(missing_codes)}")
    print("-" * 70)

    if not stations_to_process:
        print("End")
        return

    YEARS_TO_DOWNLOAD = range(2015, 2025)
    all_data_frames = []

    for year in YEARS_TO_DOWNLOAD:
        print(f"\n\n Download data by december {year}")
        print("-" * 70)

        start_of_december = datetime(year, 12, 1)
        end_of_december = datetime(year, 12, 31)

        df_december = api.download_all_stations_data(
            stations_to_process=stations_to_process,
            start_date=start_of_december,
            end_date=end_of_december
        )

        if not df_december.empty:
            all_data_frames.append(df_december)

    print("\n" + "=" * 70)
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)

        print(f"Total num of records: {len(final_df)}")

        output_filename = '../data/raw/lithuanian_weather_custom_stations.csv'
        final_df.to_csv(output_filename, index=False)
        print(f"Data saved in: {output_filename}")

    else:
        print("\nError....")


if __name__ == "__main__":
    main()