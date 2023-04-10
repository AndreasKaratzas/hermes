import os
import subprocess
import pandas as pd
import time
from tabulate import tabulate
from IPython.display import clear_output


def get_sensor_data():
    sensor_output = subprocess.check_output(
        "sensors", shell=True).decode("utf-8")
    sensor_lines = sensor_output.split("\n")

    sensor_data = []
    for line in sensor_lines:
        if "temp1" in line:
            parts = line.strip().split()
            temp_value = float(parts[1].strip("+").strip("°C"))
            sensor_data.append(temp_value)

    return sensor_data


def update_dataframe(df, sensor_data):
    new_data = pd.DataFrame(data=[sensor_data], columns=df.columns)
    df = pd.concat([df, new_data], ignore_index=True)
    return df


def print_colored_dataframe(df):
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else [0] * len(last_row)

    table = []
    for i in range(len(last_row)):
        value = last_row[i]
        color = ''
        if value > prev_row[i]:
            color = '\033[91m'  # Red
        elif value < prev_row[i]:
            color = '\033[92m'  # Green
        table.append(f"{color}{value}\033[0m")

    print(tabulate([table], headers=df.columns, tablefmt='pretty'))


def check_thresholds(df, thresholds):
    last_row = df.iloc[-1]
    for i, threshold in enumerate(thresholds):
        if last_row[i] > threshold:
            execute_another_function(i)


def execute_another_function(sensor_index):
    print(f"Sensor {sensor_index} has crossed the threshold")


# Set the temperature thresholds for each sensor
thresholds = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

# Initialize an empty DataFrame with the sensor names as columns
sensor_names = [f"Sensor {i}" for i in range(len(thresholds))]

# Initialize the DataFrame with all zeros
initial_data = [[0 for _ in range(len(thresholds))]]
df = pd.DataFrame(data=initial_data, columns=sensor_names)

while True:
    sensor_data = get_sensor_data()
    df = update_dataframe(df, sensor_data)
    clear_output(wait=True)
    print_colored_dataframe(df)
    time.sleep(5)  # Adjust the sleep interval (in seconds) between sensor readings as needed
