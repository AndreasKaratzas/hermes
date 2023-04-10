import os
import subprocess
import pandas as pd
import time
from tabulate import tabulate
import sys


def get_sensor_data():
    sensor_output = subprocess.check_output(
        "sensors", shell=True).decode("utf-8")
    sensor_lines = sensor_output.split("\n")

    sensor_values = {
        'GPU': 0,
        'LITTLE': 0,
        'BIG0': 0,
        'BIG1': 0,
        'NPU': 0,
        'CENTER': 0,
        'SOC': 0
    }

    current_sensor = ''
    for line in sensor_lines:
        if "gpu_thermal-virtual-0" in line:
            current_sensor = 'GPU'
        elif "littlecore_thermal-virtual-0" in line:
            current_sensor = 'LITTLE'
        elif "bigcore0_thermal-virtual-0" in line:
            current_sensor = 'BIG0'
        elif "npu_thermal-virtual-0" in line:
            current_sensor = 'NPU'
        elif "center_thermal-virtual-0" in line:
            current_sensor = 'CENTER'
        elif "bigcore1_thermal-virtual-0" in line:
            current_sensor = 'BIG1'
        elif "soc_thermal-virtual-0" in line:
            current_sensor = 'SOC'

        if "temp1" in line and current_sensor in sensor_values:
            sensor_values[current_sensor] = float(
                line.split('+')[1].split('°')[0])
            current_sensor = ''

    big = (sensor_values['BIG0'] + sensor_values['BIG1']) / 2
    sensor_data = [sensor_values['GPU'], sensor_values['LITTLE'], big,
                   sensor_values['NPU'], sensor_values['CENTER'], sensor_values['SOC']]

    return sensor_data


def update_dataframe(df, sensor_data):
    new_data = pd.DataFrame(data=[sensor_data], columns=df.columns)
    df = pd.concat([df, new_data], ignore_index=True)
    return df


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


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

    formatted_table = tabulate([table], headers=df.columns, tablefmt='pretty')
    clear_terminal()
    print(formatted_table)


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
    print_colored_dataframe(df)
    time.sleep(5)  # Adjust the sleep interval (in seconds) between sensor readings as needed