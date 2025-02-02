import serial
import re  # For regex matching
import threading
from utils import eeg_signal_queue

# Configure the serial connection
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

def read_eeg_data(terminate_flag):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        
        while not terminate_flag.is_set():
            if ser.in_waiting > 0:
                eeg_data = ser.readline().decode('utf-8').strip()
                print(f"Raw EEG Data: {eeg_data}")
                
                # Extract numeric value using regex (e.g., "e Peak: 0.00")
                match = re.search(r'[-+]?\d*\.\d+|\d+', eeg_data)
                if match:
                    value = float(match.group())
                    print(f"Extracted Numeric Value: {value}")
                    
                    if value >= 20:
                        print("Alpha signal detected! Posting to queue...")
                        eeg_signal_queue.put(True)
                else:
                    print("No numeric data found in the input.")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
    except ValueError as e:
        print(f"Error converting data to float: {e}")
    finally:
        if 'ser' in locals():
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    terminate_flag = threading.Event()
    try:
        read_eeg_data(terminate_flag)
    except KeyboardInterrupt:
        terminate_flag.set()
        print("Exiting...")
