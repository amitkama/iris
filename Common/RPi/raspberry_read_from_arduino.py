import serial
import time

# Replace with the correct serial port
serial_port = '/dev/ttyACM0'  # or '/dev/ttyUSB0', adjust accordingly
baud_rate = 115200
timeout = 1

# Initialize serial connection
try:
    arduino = serial.Serial(serial_port, baud_rate, timeout=timeout)
except serial.SerialException as e:
    print(f"Error opening serial port {serial_port}: {e}")
    exit(1)

time.sleep(2)  # Give Arduino time to reset

def write_read(command):
    arduino.write(command.encode('utf-8'))  # Send the 'G' command
    time.sleep(0.05)  # Small delay
    while True:
        if arduino.in_waiting > 0:  # Check if data is available
            data = arduino.readline().decode('utf-8').strip()  # Read response
            return data
        else:
            time.sleep(0.1)  # Wait a bit before checking again

# Send the command and read the data
command = 'G'
response = write_read(command)

# Print the response
if response:
    print(f"Received: {response}")
else:
    print("No data received")

# Close the serial connection
arduino.close()
