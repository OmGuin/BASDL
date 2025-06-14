import numpy as np

data = dict(np.load(r'C:\Users\omgui\Desktop\BASUS\100nm_Tetraspecs\Pre_v1p1_06Jun25-155652\Pre_v1p1Pre_12runs.par06Jun25-155652.npz', allow_pickle=True))


for key, val in data.items():
    if isinstance(val, np.ndarray):
        if val.dtype == object and val.size == 1:
            val = val.item()
        
        if isinstance(val, dict):
            print(f"* {key} (dict):")
            for subkey, subval in val.items():
                dtype = type(subval).__name__
                if isinstance(subval, (int, float, str, bool)):
                    print(f"   - {subkey}: {dtype}, value={subval}")
                else:
                    shape = np.shape(subval) if hasattr(subval, 'shape') else 'scalar'
                    print(f"   - {subkey}: {dtype}, shape={shape}")
        else:
            dtype = val.dtype.name
            shape = val.shape
            if val.shape == () or val.size == 1:
                print(f"* {key} (ndarray scalar): dtype={dtype}, value={val.item()}")
            else:
                print(f"* {key} (ndarray): dtype={dtype}, shape={shape}")
    else:
        dtype = type(val).__name__
        print(f"* {key} ({dtype}): value={val}")


#handlesA = data["handlesA"].item()
#strikeamp = handlesA["strikeamp"]
#np.save("strikeamp.npy", strikeamp)
