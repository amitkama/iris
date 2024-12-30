import ctypes
import numpy as np
from picosdk.ps6000 import ps6000 as ps
from picosdk.functions import adc2mV, assert_pico_ok
import csv
from datetime import datetime
import sys, os
import glob
import re
import datetime

receiver = sys.argv[1]
trace_num = sys.argv[2]
ct = datetime.datetime.now().time()
decimal_hour = ct.hour + ct.minute / 60
# Create chandle and status ready for use
chandle = ctypes.c_int16()
status = {}

# Open 6000 series PicoScope
status["openunit"] = ps.ps6000OpenUnit(ctypes.byref(chandle), None)
assert_pico_ok(status["openunit"])

# Set up channel A
chARange = 8  # ±5V
status["setChA"] = ps.ps6000SetChannel(chandle, 0, 1, 1, chARange, 0, 0)
assert_pico_ok(status["setChA"])

# Set up channel B
chBRange = 8  # ±5V
status["setChB"] = ps.ps6000SetChannel(chandle, 1, 1, 1, chBRange, 0, 0)
assert_pico_ok(status["setChB"])

# Set up single trigger
status["trigger"] = ps.ps6000SetSimpleTrigger(chandle, 1, 1, 64, 3, 0, 50000)
assert_pico_ok(status["trigger"])

preTriggerSamples = 5000
postTriggerSamples = 60000
maxSamples = preTriggerSamples + postTriggerSamples

timebase = 200
timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int32()
status["getTimebase2"] = ps.ps6000GetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)
assert_pico_ok(status["getTimebase2"])

status["runBlock"] = ps.ps6000RunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, 0, None, 0, None, None)
assert_pico_ok(status["runBlock"])

ready = ctypes.c_int16(0)
check = ctypes.c_int16(0)
while ready.value == check.value:
    status["isReady"] = ps.ps6000IsReady(chandle, ctypes.byref(ready))

bufferAMax = (ctypes.c_int16 * maxSamples)()
bufferBMax = (ctypes.c_int16 * maxSamples)()

status["setDataBuffersA"] = ps.ps6000SetDataBuffers(chandle, 0, ctypes.byref(bufferAMax), None, maxSamples, 0)
assert_pico_ok(status["setDataBuffersA"])

status["setDataBuffersB"] = ps.ps6000SetDataBuffers(chandle, 1, ctypes.byref(bufferBMax), None, maxSamples, 0)
assert_pico_ok(status["setDataBuffersB"])

overflow = ctypes.c_int16()
cmaxSamples = ctypes.c_int32(maxSamples)

status["getValues"] = ps.ps6000GetValues(chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow))
assert_pico_ok(status["getValues"])

maxADC = ctypes.c_int16(32512)

# Convert ADC counts data to volts
adc2VoltsChAMax = [val / 1000.0 for val in adc2mV(bufferAMax, chARange, maxADC)]  # Convert mV to V
adc2VoltsChBMax = [val / 1000.0 for val in adc2mV(bufferBMax, chBRange, maxADC)]  # Convert mV to V

time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

# Write data to CSV file

# Define the directory path
directory = f"Receiver{receiver}"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

csv_filename = f"{directory}/trace_num{trace_num}.csv"

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Time (ns)', 'Channel A (V)', 'Channel B (V)'])  # Write header
    for i in range(len(time)):
        csv_writer.writerow([time[i], adc2VoltsChAMax[i], adc2VoltsChBMax[i]])

status["stop"] = ps.ps6000Stop(chandle)
assert_pico_ok(status["stop"])

ps.ps6000CloseUnit(chandle)

print(status)
