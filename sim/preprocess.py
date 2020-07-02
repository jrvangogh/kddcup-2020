"""Preprocess data for use in simulation.

The following files should be placed in data/raw:
- gps_01-10.zip
- gps_11-20.zip
- gps_21-30.zip
- order_01-30.zip
"""
import csv
import sys
from dataclasses import asdict, dataclass
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

ORDER_FILE = 'order.zip'
GPS_FILES = ['trajectory_1.zip', 'trajectory_2.zip', 'trajectory_3.zip']

ORDER_FILE_COLUMNS = [
    'order_id',
    'start_time',
    'stop_time',
    'pickup_lng',
    'pickup_lat',
    'dropoff_lng',
    'dropoff_lat',
    'reward']

CANCEL_FILE_COLUMNS = ['order_id'] + [f'cancel_prob_{r}m' for r in range(200, 2001, 200)]


@dataclass
class DriverSchedule:
    driver_id: str
    start_lat: float = 0.0
    start_lng: float = 0.0
    start_time: int = sys.maxsize
    end_time: int = -1


def process_trajectory_file(f, output_file_dir):
    """Build driver schedules from trajectory file and save to disk.

    This is done in a streaming fashion to avoid loading entire trajectory files into memory at once.
    """
    ds = f.name.split('_')[1]

    drivers = dict()

    for line in f:
        driver_id, order_id, timestamp, lng, lat = line.decode().split(',')
        timestamp = int(timestamp)

        # Get driver
        if driver_id not in drivers:
            drivers[driver_id] = DriverSchedule(driver_id)
        d = drivers[driver_id]

        # Update driver schedule
        if d.start_time is None or d.start_time > timestamp:
            d.start_time = timestamp
            d.start_lat = float(lat)
            d.start_lng = float(lng)
        d.end_time = max(d.end_time, timestamp)

    # Data filters
    print(f'{len(drivers)} drivers processed for {ds}')
    drivers = {driver_id: driver_schedule for driver_id, driver_schedule in drivers.items() if
               driver_schedule.end_time > driver_schedule.start_time}
    print(f'{len(drivers)} after filtering zero length shifts for {ds}')

    output_file_full_path = output_file_dir / 'drivers'
    output_file_full_path.mkdir(exist_ok=True)

    df = pd.DataFrame([asdict(d) for d in drivers.values()])
    df = df.sort_values(by='start_time', ignore_index=True)

    df.to_parquet(output_file_full_path / f'{ds}.parquet', index=False)


def process_order_files(interim_data_path, processed_data_path):
    """Process all order files, combining orders with cancel probabilities."""
    output_file_path = (processed_data_path / 'orders')
    output_file_path.mkdir(exist_ok=True)

    order_file_iterator = (interim_data_path / 'total_ride_request').iterdir()

    for order_file in order_file_iterator:
        df_order = pd.read_csv(order_file, names=ORDER_FILE_COLUMNS)
        cancel_file = interim_data_path / 'total_order_cancellation_probability' / f'{order_file.name}_cancel_prob'
        df_cancel = pd.read_csv(cancel_file, names=CANCEL_FILE_COLUMNS)

        df_cancel['cancel_prob'] = df_cancel[df_cancel.columns[1:]].values.tolist()
        df_cancel = df_cancel[['order_id', 'cancel_prob']]

        df = df_order.merge(df_cancel, on='order_id')
        df.sort_values(by='start_time', ignore_index=True)

        df.to_parquet(output_file_path / f'{order_file.name.split("_")[1]}.parquet', index=False)
        print(f'{order_file.name} written')


def main():
    project_path = Path('/mnt/user-home/kddcup-2020')

    raw_data_path = project_path / 'zip_data'
    interim_data_path = project_path / 'interim_data'
    processed_data_path = project_path / 'processed_data'

    interim_data_path.mkdir(exist_ok=True)
    processed_data_path.mkdir(exist_ok=True)

    # Extract orders zip
    if not (interim_data_path / 'hexagon_grid_table.csv').exists():
        with ZipFile(raw_data_path / ORDER_FILE) as z:
            z.extractall(interim_data_path)

    # Process order files
    process_order_files(interim_data_path, processed_data_path)

    # Process trajectories
    for gps_file in GPS_FILES:
        with ZipFile(raw_data_path / gps_file) as z:
            for name in z.namelist():
                with z.open(name, 'r') as f:
                    process_trajectory_file(f, processed_data_path)


if __name__ == '__main__':
    main()
