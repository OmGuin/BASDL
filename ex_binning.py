import numpy as np
import matplotlib.pyplot as plt


filename = r'C:\Users\omgui\Desktop\BASUS\real_data\100nm_Tetraspecs\100nm_TetraSpec_FS_1.txt' 


with open(filename) as f:
    for i, line in enumerate(f):
        if line.startswith('Ch 1'):
            header_lines = i + 1
            break


ch1_data = np.loadtxt(filename, skiprows=header_lines, usecols=0)


conversion_factor = 82.31e-12
times_seconds = ch1_data * conversion_factor
times_seconds = times_seconds[times_seconds>0]

bin_size = 500e-6
max_time = np.max(times_seconds)
min_time = np.min(times_seconds)
num_bins = int(np.ceil((max_time - min_time) / bin_size))
bin_edges = np.linspace(min_time, max_time, num_bins + 1)


plt.figure(figsize=(10, 6))
plt.hist(times_seconds, bins=bin_edges, 
         edgecolor='black', color='blue', alpha=0.7)


plt.title('Photon Arrival Times Histogram (500 μs bins)')
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()