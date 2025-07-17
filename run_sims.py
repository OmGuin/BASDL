#imports
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
from datetime import datetime
import json
from numba import njit, prange
import multiprocessing as mp
#directory
data_dir = "PCHdataset_LCjawn"
otherdata_dir = "diff_testing"
os.makedirs(data_dir, exist_ok=True)
D=1e-11
totaltime = 60
binDt = 5e-6
w0 = 3e-7
axialFactor=3
vFlow = 5e-4
def SimPhotDiffFlowGL6(C_molar, Rp, D, totalTime, binDt, w0, axialFactor, includeBg, bgRate, beamShape, vFlow, rng, resFactor=10):
    # 1) Geometry & Constraints
    NA = 6.022e23
    C_m3 = C_molar * 1e3
    w_z = axialFactor * w0
    Lbox = 5 * max(w0, w_z)
    Lres = resFactor * Lbox

    # 2) Reservoir initialization
    area_yz = (2*Lbox)**2
    Vres = (2*Lres) * area_yz
    Nres = max(1, rng.poisson(C_m3 * NA * Vres))
    pos = np.empty((Nres, 3))
    pos[:, 0] = (rng.random(Nres)-0.5)*2*Lres
    pos[:, 1] = (rng.random(Nres)-0.5)*2*Lbox
    pos[:, 2] = (rng.random(Nres)-0.5)*2*Lbox

    # 3) Time step and diffusion
    dt = binDt
    sigma = np.sqrt(2 * D * dt)
    nSteps = int(np.ceil(totalTime/dt))
    if vFlow > 0:
        stepsPerSweep = int(np.ceil((2*Lres) / (vFlow * dt)))
    else:
        stepsPerSweep = int(1e9)

    # 4) Preallocate photon times
    Veff = np.pi ** (3/2) * w0**2 * w_z
    Navg = C_m3 * NA * Veff
    expCount = int(np.ceil((Navg*Rp + bgRate) * totalTime * 1.2))
    sigma_b = 0.2
    Rp_i = np.ones(Nres) * Rp#np.exp(rng.normal(loc=np.log(Rp), scale=sigma_b, size=Nres))
    # Pre-generate all random numbers needed for the loop
    #diffusion_noise = rng.standard_normal((nSteps, Nres, 3))
    
    photon_uniform = rng.random((nSteps, int(2*Navg*Rp*dt+10)))  # overestimate
    print(photon_uniform.shape)
    poisson_photons = rng.poisson(1.0, nSteps)  # will scale by mean in loop
    poisson_bg = rng.poisson(1.0, nSteps) if includeBg else np.zeros(nSteps)
    perm_indices = rng.integers(0, Nres, (nSteps//stepsPerSweep+2, Nres))
    

    # Call the JIT-compiled simulation loop
    arrivalTimes, idx, Rp_i_out = simulation_loop_jit(
        pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
        includeBg, bgRate, beamShape, photon_uniform, poisson_photons, poisson_bg, perm_indices
    )
    arrivalTimes = arrivalTimes[:idx]
    # 7) Bin into intensity trace
    edges = np.arange(0, totalTime, binDt)
    counts, _ = np.histogram(arrivalTimes, bins=edges)
    timeBins = edges[:-1] + binDt/2
    return arrivalTimes, counts, timeBins, Rp_i_out

@njit
def simulation_loop_jit(pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
                       includeBg, bgRate, beamShape, photon_uniform, poisson_photons, poisson_bg, perm_indices):
    arrivalTimes = np.empty(nSteps*100, dtype=np.float64)  # overallocate
    idx = 0
    perm_counter = 0
    for k in range(1, nSteps+1):
        t0 = (k-1) * dt
        # Advect
        pos[:, 0] += vFlow * dt
        pos[:, 0] = np.mod(pos[:, 0] + Lres, 2*Lres) - Lres
        # Diffuse
        inBox = np.abs(pos[:, 0]) <= Lbox
        n_inBox = np.sum(inBox)
        #if n_inBox > 0:
        #    pos[inBox, :] += sigma * diffusion_noise[k-1, inBox, :]
        # Reflect boundaries
        for i in range(Nres):
            if inBox[i]:
                for j in range(3):
                    pos[i, j] += sigma * np.random.standard_normal()
                if pos[i, 0] > Lbox:
                    pos[i, 0] = 2*Lbox - pos[i, 0]
                elif pos[i, 0] < -Lbox:
                    pos[i, 0] = -2*Lbox - pos[i, 0]
                if pos[i, 1] > Lbox:
                    pos[i, 1] = 2*Lbox - pos[i, 1]
                elif pos[i, 1] < -Lbox:
                    pos[i, 1] = -2*Lbox - pos[i, 1]
                if pos[i, 2] > Lbox:
                    pos[i, 2] = 2*Lbox - pos[i, 2]
                elif pos[i, 2] < -Lbox:
                    pos[i, 2] = -2*Lbox - pos[i, 2]
        # Photon emission
        xy2 = pos[:, 0]**2 + pos[:, 1]**2
        z2 = pos[:, 2]**2
        if beamShape == 'gaussian':
            W = np.exp(-2 * xy2/w0**2 - 2*z2/w_z**2)
        else:
            Wlat = np.exp(-2*xy2/w0**2)
            Wax = 1 / (1 + z2/w_z**2)
            W = Wlat * Wax
        Rtot = np.sum(Rp_i * W)
        mean_photons = Rtot * dt
        Nph = int(poisson_photons[k-1] * mean_photons)
        Nbg = int(poisson_bg[k-1] * bgRate * dt) if includeBg else 0
        NtotEv = Nph + Nbg
        if NtotEv > 0:
            for j in range(NtotEv):
                arrivalTimes[idx] = t0 + photon_uniform[k-1, j] * dt
                idx += 1
        # Permute
        if stepsPerSweep < 1e8 and k % stepsPerSweep == 0:
            perm = perm_indices[perm_counter]
            for d in range(3):
                pos[:, d] = pos[perm, d]
            perm_counter += 1
    return arrivalTimes, idx, Rp_i

def run_one_sim(sim_num, conc=9e-12, amp=2000):
    seed = sim_num + int(time.time()) % 10000
    rng = np.random.default_rng(seed)
    start_sim = time.time()

    sim_dir = os.path.join(data_dir, f"sim_{sim_num:07}")
    os.makedirs(sim_dir, exist_ok=True)


    num_species = 1#np.random.randint(1,4) #[1,3]
    AmpS1 = amp#rng.integers(1000, 3000)
    AmpS2 = rng.integers(AmpS1+1500, AmpS1+1500+2000)
    AmpS3 = rng.integers(AmpS2+1500, AmpS2+1500+2500)
    total_conc = conc#rng.uniform(9e-12, 7e-11)
    min_frac = 0.15

    while True:
        fractions = rng.dirichlet([1.5] * num_species)
        if np.all(fractions > min_frac):
            break

    Frac1 = fractions[0]
    Frac2 = fractions[1] if num_species >= 2 else 0.0
    Frac3 = fractions[2] if num_species == 3 else 0.0
    conc1 = Frac1 * total_conc
    conc2 = Frac2 * total_conc
    conc3 = Frac3 * total_conc

    truedist1 = np.array([])
    truedist2 = np.array([])
    truedist3 = np.array([])
    at1 = np.array([])
    at2 = np.array([])
    at3 = np.array([])


    arrivalTimes1, counts, timeBins, Rp_i1 = SimPhotDiffFlowGL6(C_molar = conc1,
                                                        Rp = AmpS1/500e-6, 
                                                        D = D, 
                                                        totalTime = totaltime, 
                                                        binDt = binDt, 
                                                        w0 = w0, 
                                                        axialFactor = axialFactor, 
                                                        includeBg = True, 
                                                        bgRate = 1e2, 
                                                        beamShape = 'gl',
                                                        vFlow=vFlow,
                                                        rng = rng)
    truedist1 = Rp_i1
    at1 = arrivalTimes1
    if num_species >= 2:
        arrivalTimes2, counts, timeBins, Rp_i2 = SimPhotDiffFlowGL6(C_molar = conc2,
                                                            Rp = AmpS2/500e-6, 
                                                            D = D, 
                                                            totalTime = totaltime, 
                                                            binDt = binDt, 
                                                            w0 = w0, 
                                                            axialFactor = axialFactor, 
                                                            includeBg = False, 
                                                            bgRate = 0, 
                                                            beamShape = 'gl',
                                                            vFlow=vFlow,
                                                            rng = rng)
        truedist2 = Rp_i2
        at2 = arrivalTimes2
    if num_species == 3:
        arrivalTimes3, counts, timeBins, Rp_i3 = SimPhotDiffFlowGL6(C_molar = conc3,
                                                            Rp = AmpS3/500e-6, 
                                                            D = D, 
                                                            totalTime = totaltime, 
                                                            binDt = binDt, 
                                                            w0 = w0, 
                                                            axialFactor = axialFactor, 
                                                            includeBg = False, 
                                                            bgRate = 0, 
                                                            beamShape = 'gl',
                                                            vFlow=vFlow,
                                                            rng = rng)
        truedist3 = Rp_i3
        at3 = arrivalTimes3

    fullBrightDist = np.concatenate([truedist1.flatten()*500e-6, truedist2.flatten()*500e-6, truedist3.flatten()*500e-6])
    bins = np.linspace(0, 12000, 21)
    truedist, bin_edges_true_dist = np.histogram(fullBrightDist, bins = bins)
    # plt.bar(bin_edges_true_dist[:-1], truedist, width=np.diff(bin_edges_true_dist), edgecolor='black', align='edge')
    # plt.title('Underlying Brightness Distribution', fontsize=14)
    # plt.xlabel('Fluorescence Intensity', fontsize=12)
    # plt.ylabel('Number of particles', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)

    fullTOAs = np.concatenate([at1, at2, at3])
    fullTOAs = np.sort(fullTOAs)

    bins_hist = np.linspace(0, totaltime, int((totaltime)/(500e-6)) + 1)
    histA, edges = np.histogram(fullTOAs, bins_hist)
    PCHedges = np.linspace(0, 12000, 101) 
    PCHbins, _ = np.histogram(histA, bins=PCHedges)
    
    

    GT = {
        "Amplitudes":{
            "AmpS1":int(AmpS1),
            "AmpS2":int(AmpS2),
            "AmpS3":int(AmpS3)
        },
        "ActualFractions": {
            "Frac1": float(Frac1),
            "Frac2": float(Frac2),
            "Frac3": float(Frac3)
        },
        "ActualConcentrations":{
            "Species1":float(conc1),
            "Species2":float(conc2),
            "Species3":float(conc3),
            "Total":float(total_conc)
        },
        "SimulationInputs":{
            "D":D,
            "totaltime":totaltime,
            "binDt":binDt,
            "w0":w0,
            "axialFactor":axialFactor,
            "vFlow":vFlow
        },
        "Other":{
            "Seed":seed
        }
    }

    with open(os.path.join(sim_dir, "GT.json"), "w") as f:
        json.dump(GT, f)

    np.save(os.path.join(sim_dir, "true_bins.npy"), truedist)
    np.save(os.path.join(sim_dir, "true_edges.npy"), bin_edges_true_dist)
    np.save(os.path.join(sim_dir, "pchbins.npy"), PCHbins)
    np.save(os.path.join(sim_dir, "pchedges.npy"), PCHedges)
    savemat(os.path.join(otherdata_dir, f"arrivalTimes{conc}_{amp}.mat"), {"arrivalTimes": fullTOAs.reshape(-1, 1)})
    #np.savez_compressed(os.path.join(sim_dir, "TOAs.npy"), TOAs = fullTOAs)
    print(f"Finished sim {sim_num}, Time: {time.time() - start_sim}")
    return True
# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)

#     num_sims = 100
#     num_workers = 8

#     with mp.Pool(processes=num_workers) as pool:
#         pool.map(run_one_sim, range(num_sims))

concs = [9e-12, 4e-11, 7e-11]
amps = [1000, 5250, 10500]

start = time.time()
k = 0
for conc in concs:
    for amp in amps:
        run_one_sim(k, conc, amp)
        k+=1

print((time.time() - start)/10)