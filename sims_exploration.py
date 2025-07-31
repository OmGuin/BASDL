#imports
import time
import numpy as np
from scipy.io import savemat
import os
import json
from numba import njit
import multiprocessing as mp
#directory
data_dir = "PCHdataset_LCjawn"
otherdata_dir = "720_data"
os.makedirs(otherdata_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
w0 = 3e-7
axialFactor=3
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
    sigma_b = 0
    #Rp_i = np.exp(rng.normal(loc=np.log(Rp), scale=sigma_b, size=Nres))
    b = np.exp(sigma_b * rng.normal(size=Nres) - 0.5 * sigma_b**2)
    Rp_i = Rp * b
    # Pre-generate all random numbers needed for the loop
    #diffusion_noise = rng.standard_normal((nSteps, Nres, 3))
    photon_uniform = rng.random((nSteps, int(2*Rp*dt+100)))  # overestimate
    poisson_photons = rng.poisson(1.0, nSteps)  # will scale by mean in loop
    #poisson_bg = rng.poisson(1.0, nSteps) if includeBg else np.zeros(nSteps)
    poisson_bg = rng.poisson(bgRate * dt, nSteps) if includeBg else np.zeros(nSteps, dtype=np.int64)

    # perm_indices = rng.integers(0, Nres, (nSteps//stepsPerSweep+2, Nres))
    perm_indices_x = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))
    perm_indices_y = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))
    perm_indices_z = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))


    # Call the JIT-compiled simulation loop
    arrivalTimes, idx, Rp_i_out = simulation_loop_jit(
        pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
        includeBg, bgRate, beamShape, photon_uniform, poisson_photons, poisson_bg, perm_indices_x, perm_indices_y, perm_indices_z
    )
    arrivalTimes = arrivalTimes[:idx]
    # 7) Bin into intensity trace
    edges = np.arange(0, totalTime, binDt)
    counts, _ = np.histogram(arrivalTimes, bins=edges)
    timeBins = edges[:-1] + binDt/2
    return arrivalTimes, counts, timeBins, Rp_i_out
@njit
def simulation_loop_jit(pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
                       includeBg, bgRate, beamShape, photon_uniform, poisson_photons, poisson_bg, perm_indices_x, perm_indices_y, perm_indices_z):
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
        
        # for i in range(Nres):
        #     if inBox[i]:
        #         for j in range(3):
        #             pos[i, j] += sigma * np.random.standard_normal()
        #         if pos[i, 0] > Lbox:
        #             pos[i, 0] = 2*Lbox - pos[i, 0]
        #         elif pos[i, 0] < -Lbox:
        #             pos[i, 0] = -2*Lbox - pos[i, 0]
        #         if pos[i, 1] > Lbox:
        #             pos[i, 1] = 2*Lbox - pos[i, 1]
        #         elif pos[i, 1] < -Lbox:
        #             pos[i, 1] = -2*Lbox - pos[i, 1]
        #         if pos[i, 2] > Lbox:
        #             pos[i, 2] = 2*Lbox - pos[i, 2]
        #         elif pos[i, 2] < -Lbox:
        #             pos[i, 2] = -2*Lbox - pos[i, 2]

        for i in range(Nres):
            if inBox[i]:
                # Apply diffusion step
                for d in range(3):
                    pos[i, d] += sigma * np.random.standard_normal()

                # Reflect in each axis
                if pos[i, 0] > Lbox:
                    pos[i, 0] = 2 * Lbox - pos[i, 0]
                elif pos[i, 0] < -Lbox:
                    pos[i, 0] = -2 * Lbox - pos[i, 0]
                if pos[i, 1] > Lbox:
                    pos[i, 1] = 2 * Lbox - pos[i, 1]
                elif pos[i, 1] < -Lbox:
                    pos[i, 1] = -2 * Lbox - pos[i, 1]
                if pos[i, 2] > Lbox:
                    pos[i, 2] = 2 * Lbox - pos[i, 2]
                elif pos[i, 2] < -Lbox:
                    pos[i, 2] = -2 * Lbox - pos[i, 2]
        
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
        Nph = np.random.poisson(mean_photons)#int(poisson_photons[k-1] * mean_photons)
        #Nbg = int(poisson_bg[k-1] * bgRate * dt) if includeBg else 0
        Nbg = poisson_bg[k-1] if includeBg else 0
        NtotEv = Nph + Nbg
        if NtotEv > 0:
            for j in range(NtotEv):
                if idx >= len(arrivalTimes) or NtotEv > photon_uniform.shape[1]:
                    print("underallocated mem")
                
                arrivalTimes[idx] = t0 + photon_uniform[k-1, j] * dt

                idx += 1
        # Permute
        if stepsPerSweep < 1e8 and k % stepsPerSweep == 0:
            # perm = perm_indices[perm_counter]
            perm_x = perm_indices_x[perm_counter]
            perm_y = perm_indices_y[perm_counter]
            perm_z = perm_indices_z[perm_counter]
            pos[:, 0] = pos[perm_x, 0]
            pos[:, 1] = pos[perm_y, 1]
            pos[:, 2] = pos[perm_z, 2]
            perm_counter += 1
    return arrivalTimes, idx, Rp_i

def run_one_sim(sim_num, D, dt, conc, Rp, vF, tt):
    seed = sim_num + int(time.time()) % 10000
    rng = np.random.default_rng(seed)
    start_sim = time.time()

    sim_dir = os.path.join(data_dir, f"sim_{sim_num:07}")
    os.makedirs(sim_dir, exist_ok=True)


    num_species = 1#np.random.randint(1,4) #[1,3]
    AmpS1 = Rp * 500e-6#rng.integers(1000, 3000)
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


    arrivalTimes1, counts, timeBins, Rp_i1 = SimPhotDiffFlowGL6(C_molar = conc,
                                                        Rp = Rp, 
                                                        D = D, 
                                                        totalTime = tt, 
                                                        binDt = dt, 
                                                        w0 = w0, 
                                                        axialFactor = axialFactor, 
                                                        includeBg = False, 
                                                        bgRate = 0, 
                                                        beamShape = 'gl',
                                                        vFlow=vF,
                                                        rng = rng)
    truedist1 = Rp_i1
    at1 = arrivalTimes1
    if num_species >= 2:
        arrivalTimes2, counts, timeBins, Rp_i2 = SimPhotDiffFlowGL6(C_molar = conc2,
                                                            Rp = AmpS2/500e-6, 
                                                            D = D, 
                                                            totalTime = 10, 
                                                            binDt = 1e-6, 
                                                            w0 = w0, 
                                                            axialFactor = axialFactor, 
                                                            includeBg = False, 
                                                            bgRate = 0, 
                                                            beamShape = 'gl',
                                                            vFlow=5e-4,
                                                            rng = rng)
        truedist2 = Rp_i2
        at2 = arrivalTimes2
    if num_species == 3:
        arrivalTimes3, counts, timeBins, Rp_i3 = SimPhotDiffFlowGL6(C_molar = conc3,
                                                            Rp = AmpS3/500e-6, 
                                                            D = D, 
                                                            totalTime = 10, 
                                                            binDt = 1e-6, 
                                                            w0 = w0, 
                                                            axialFactor = axialFactor, 
                                                            includeBg = False, 
                                                            bgRate = 0, 
                                                            beamShape = 'gl',
                                                            vFlow=5e-4,
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
    #{datetime.datetime.now().strftime("%m_%d_%H_%M")}
    bins_hist = np.linspace(0, tt, int((tt)/(500e-6)) + 1)
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
            "totaltime":tt,
            "binDt":dt,
            "w0":w0,
            "axialFactor":axialFactor,
            "vFlow":vF
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
    
    savemat(os.path.join(otherdata_dir, f"arrivalTimes_{D:.1e}_{dt:.1e}_{conc:.1e}_{Rp:.1e}_{vF:.1e}_{tt}_{sim_num}.mat"), {"arrivalTimes": fullTOAs.reshape(-1, 1)})
    #np.savez_compressed(os.path.join(sim_dir, "TOAs.npy"), TOAs = fullTOAs)
    print(f"Finished sim {sim_num}, Time: {time.time() - start_sim}")
    return 
# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)

#     num_sims = 100
#     num_workers = 8

#     with mp.Pool(processes=num_workers) as pool:
#         pool.map(run_one_sim, range(num_sims))


# #1

# d: 4e-10, 1e-10, 7e-11, 3e-11, 8e-12, 4e-12
# conc: 1e-10, 5e-11, 1e-11, 5e-12, 1e-12
# rp: 1e5, 1e6, 1e7
# vflow: 5e-3, 0
# dt: 1e-6, 1e-5

# #1
# d = 4e-10
# dt = 1e-6
# conc = 5e-12
# rp = 1e6
# vFlow = 5e-4

# for each D:
#     for each dt:
#         conc = 5e-12, Rp = 1e6, vF = 5e-4
# for each C:
#     D = 7e-11, Rp = 1e6, vF = 5e-4, dt = 1e-6
# for each Rp:
#     D = 7e-11, conc = 5e-12, vF = 5e-4, dt = 1e-6
# for each vF:
#     D = 7e-11, Rp=1e6, dt = 1e-6, conc = 5e-12
        

# if __name__ == "__main__":
#     Ds = [4e-10, 1e-10, 7e-11, 3e-11, 8e-12, 4e-12]
#     Cs = [1e-10, 5e-11, 1e-11, 5e-12, 1e-12]
#     dts = [1e-6, 1e-5]
#     Rps = [1e5, 1e6, 1e7]
#     vFs = [5e-4, 0]
#     dts = [1e-6, 1e-5]
#     combos = [] #D, dt, conc, rp, vF
#     for D in Ds:
#         for dt in dts:
#             combos.append((D, dt, 5e-12, 1e6, 5e-4, 10))
#     for C in Cs:
#         combos.append((7e-11, 1e-6, C, 1e6, 5e-4, 10))
#     for Rp in Rps:
#         combos.append((7e-11, 1e-6, 5e-12, Rp, 5e-4, 10))
#     for vF in vFs:
#         combos.append((7e-11, 1e-6, 5e-12, 1e6, vF, 10))
#     combos.append((7e-11, 1e-6, 5e-12, 1e6, 5e-4, 60))
#     combos = [(i, D, dt, conc, Rp, vF, tt) for i, (D, dt, conc, Rp, vF, tt) in enumerate(combos)]

    
#     with mp.Pool(processes=4) as pool:
#         results = pool.starmap(run_one_sim, combos)