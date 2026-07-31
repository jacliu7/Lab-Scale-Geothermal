import time
import glob

# Base path where 1-Wire devices show up
BASE_DIR = '/sys/bus/w1/devices/'

def find_sensors():
    """Auto-detect all connected DS18B20 sensors."""
    device_folders = glob.glob(BASE_DIR + '28*')
    if len(device_folders) < 2:
        raise RuntimeError(f"Expected 2 sensors, found {len(device_folders)}. "
                            "Check wiring and that 1-Wire is enabled.")
    return device_folders

def read_temp_raw(device_file):
    with open(device_file, 'r') as f:
        return f.readlines()

def read_temp(device_folder):
    device_file = device_folder + '/w1_slave'
    lines = read_temp_raw(device_file)

    # Retry if the sensor's CRC check failed (line doesn't end in "YES")
    retries = 0
    while lines[0].strip()[-3:] != 'YES' and retries < 10:
        time.sleep(0.2)
        lines = read_temp_raw(device_file)
        retries += 1

    equals_pos = lines[1].find('t=')
    if equals_pos == -1:
        return None

    temp_string = lines[1][equals_pos + 2:]
    temp_c = float(temp_string) / 1000.0
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    return temp_c, temp_f

def main():
    sensors = find_sensors()
    print(f"Found {len(sensors)} sensor(s):")
    for s in sensors:
        print(f"  {s.split('/')[-1]}")
    print()

    try:
        while True:
            for sensor_folder in sensors:
                sensor_id = sensor_folder.split('/')[-1]
                result = read_temp(sensor_folder)
                if result:
                    c, f = result
                    print(f"Sensor {sensor_id}: {c:.2f}°C / {f:.2f}°F")
                else:
                    print(f"Sensor {sensor_id}: read error")
            print('-' * 40)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()