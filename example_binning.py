import numpy as np
import matplotlib.pyplot as plt


filename = r'C:\Users\omgui\Desktop\BASDL\real_data\100nm_Tetraspecs\100nm_TetraSpec_FS_1.txt' 


with open(filename) as f:
    for i, line in enumerate(f):
        if line.startswith('Ch 1'):
            header_lines = i + 1
            break


ch1_data = np.loadtxt(filename, skiprows=header_lines, usecols=0)

conversion_factor = 82.31e-12
times_seconds = ch1_data * conversion_factor
times_seconds = times_seconds[times_seconds > 0]

tbin = 500e-6
Tottime = 5  # seconds
Nafterbin = round(Tottime / tbin)
tedge = np.arange(0, Nafterbin + 1) * tbin

histA, _ = np.histogram(times_seconds, bins=tedge)
I_A = histA.tolist()  # or use extend() in a loop if you're concatenating across files

d = np.array(I_A)
time_vector = np.arange(len(d)) * tbin

handles = {
    'd': np.vstack([time_vector, d]),
}

plt.figure(figsize=(12, 8))
plt.plot(handles['d'][0, :], handles['d'][1, :], 'b', label='Counts')
plt.plot(handles['d'][0, :], handles['d'][1, :], '.r')
plt.title("Intensity vs Time", fontsize=12)
plt.xlabel("Time (seconds)")
plt.ylabel("Photon counts per time bin")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()