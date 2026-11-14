"""Run: python diag.py --port COM5
Shows what the ESP32 is sending to the laptop for 5 seconds."""
import argparse
import time

import serial

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True)
args = ap.parse_args()

ser = serial.Serial(args.port, 921600, timeout=0.1)
time.sleep(2)
ser.reset_input_buffer()
ser.write(bytes([0xA5, 0x35, 0x02, 0xDC]))   # ask for automatic output

data = bytearray()
t0 = time.time()
while time.time() - t0 < 5:
    data += ser.read(4096)

print(f"Received {len(data)} bytes in 5 s")
print("First 32 bytes:", data[:32].hex(" ") if data else "(none)")
print("Header 5A 5A found:", b"\x5a\x5a" in data)
print("Approx frames (1544 bytes each):", len(data) // 1544)

if not data:
    print("-> NO DATA: check wiring (swap TX/RX), 3.3V power, and that the SET I2C jumper is open.")
elif b"\x5a\x5a" not in data:
    print("-> DATA BUT NO HEADER: wrong baud. Change SENSOR_BAUD to 9600 in the sketch and re-upload.")
else:
    print("-> Sensor is working. The viewer should display.")
