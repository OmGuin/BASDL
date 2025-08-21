import numpy as np
import matplotlib.pyplot as plt
import json

with open(r"C:\Users\omgui\Desktop\BASDL\data_gen_phase1a\pchdata\sim_00009\GT.json", 'r') as f:
    jawn = json.load(f)

tdist = np.array(jawn['Data']['true_bins'])
pch = np.array(jawn['Data']['pch_bins'])

tdist2 = np.array(jawn['Data']['true_bins'])
pch2 = np.array(jawn['Data']['pch_bins'])

print(sum(pch))
print(sum(tdist))

# tdist = np.load(r"C:\Users\omgui\Desktop\BASDL\data_gen_phase1a\pchdata\sim_00003\true_bins (1).npy")
# pch = np.load(r"C:\Users\omgui\Desktop\BASDL\data_gen_phase1a\pchdata\sim_00003\pchbins.npy")

# tdist2 = np.load(r"C:\Users\omgui\Desktop\BASDL\data_gen_phase1a\pchdata\sim_00004\true_bins (1).npy")
# pch2 = np.load(r"C:\Users\omgui\Desktop\BASDL\data_gen_phase1a\pchdata\sim_00004\pchbins.npy")


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].plot(pch, marker='o', label='True PCH')
# axes[0].set_xlabel("Bin")
# axes[0].set_ylabel("Intensity Count (#)")
# axes[0].legend()
# axes[0].grid(True)


axes[0].plot(tdist, marker='o', label='True Brightness')
axes[0].set_xlabel("Bin")
axes[0].set_ylabel("Probability")
#axes[0].set_ylim((0, 1))
axes[0].legend()
axes[0].grid(True)


axes[1].plot(pch, marker='o', label='PCH')
axes[1].set_xlabel("Bin")
axes[1].set_ylabel("Probability")
#axes[1].set_ylim((0, 1))
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
