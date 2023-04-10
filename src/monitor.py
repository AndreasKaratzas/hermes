import os
import subprocess
import pandas as pd
import time
from io import StringIO
from IPython.display import clear_output, display


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


def style_dataframe(df):
    def highlight_changes(s):
        colors = []
        for i in range(len(s)):
            if i == 0:
                colors.append('background-color: red')
            elif s.iloc[i] > s.iloc[i - 1]:
                colors.append('background-color: red')
            elif s.iloc[i] < s.iloc[i - 1]:
                colors.append('background-color: green')
            else:
                colors.append('')
        return colors

    styled_df = df.style.apply(highlight_changes)
    return styled_df


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
initial_data = [[0 for _ in range(len(thresholds))]]
df = pd.DataFrame(data=initial_data, columns=sensor_names)

while True:
    sensor_data = get_sensor_data()
    df = update_dataframe(df, sensor_data)
    styled_df = style_dataframe(df)
    clear_output(wait=True)
    display(styled_df)
    time.sleep(5)  # Adjust the sleep interval (in seconds) between sensor readings as needed
