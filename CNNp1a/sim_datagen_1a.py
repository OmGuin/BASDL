#imports
import time
import numpy as np
import os
import json
from numba import njit
import multiprocessing as mp
import gc

#directory
data_dir = os.path.join("data_gen_phase1a", "pchdata")
os.makedirs(data_dir, exist_ok=True)
w0 = 3e-7
axialFactor=3

def SimPhotDiffFlowGL6(C_molar, Rp, D, totalTime, binDt, w0, axialFactor, includeBg, bgRate, beamShape, vFlow, rng, num_species, resFactor=10):
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
    alpha = None
    
    sigma_b = None
    dist_type = 1
    if num_species == 1:
        dist_type = np.random.randint(0, 2)
        if dist_type == 0: # power_law
            alpha = rng.uniform(1.0, 2.5)
            bmin, bmax = 0.2, 4.2
            u = rng.random(Nres)
            if np.isclose(alpha, 1.0, atol=1e-6):
                b = bmin * (bmax / bmin) ** u
            else:
                p = 1.0 - alpha
                b = (u * (bmax**p - bmin**p) + bmin**p) ** (1.0 / p)
            b = b / b.mean()
        else: # log normal
            sigma_b = np.random.uniform(0, 0.2)
            #Rp_i = np.exp(rng.normal(loc=np.log(Rp), scale=sigma_b, size=Nres))
            b = np.exp(sigma_b * rng.normal(size=Nres) - 0.5 * sigma_b**2)
    else:
        sigma_b = np.random.uniform(0, 0.2)
        #Rp_i = np.exp(rng.normal(loc=np.log(Rp), scale=sigma_b, size=Nres))
        b = np.exp(sigma_b * rng.normal(size=Nres) - 0.5 * sigma_b**2)

    Rp_i = Rp * b
    
    # Pre-generate all random numbers needed for the loop
    #diffusion_noise = rng.standard_normal((nSteps, Nres, 3))
    
    #photon_uniform = rng.random((nSteps, int(2*Rp*dt+100)))  # overestimate
    #poisson_photons = rng.poisson(1.0, nSteps)  # will scale by mean in loop
    
    #poisson_bg = rng.poisson(1.0, nSteps) if includeBg else np.zeros(nSteps)
    poisson_bg = rng.poisson(bgRate * dt, nSteps) if includeBg else np.zeros(nSteps, dtype=np.int64)

    # perm_indices = rng.integers(0, Nres, (nSteps//stepsPerSweep+2, Nres))
    perm_indices_x = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))
    perm_indices_y = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))
    perm_indices_z = rng.integers(0, Nres, (nSteps//stepsPerSweep + 2, Nres))

    # 4) Preallocate photon times
    Veff = np.pi ** (3/2) * w0**2 * w_z
    Navg = C_m3 * NA * Veff
    expCount = int(np.ceil((Navg*Rp + bgRate) * totalTime * 3.0))

    arrivalTimes = np.empty(expCount, dtype=np.float64)

    # Call the JIT-compiled simulation loop
    idx, Rp_i_out = simulation_loop_jit(
        arrivalTimes,
        pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
        includeBg, bgRate, beamShape, poisson_bg, perm_indices_x, perm_indices_y, perm_indices_z
    )
    arrivalTimes = arrivalTimes[:idx]
    # 7) Bin into intensity trace
    edges = np.arange(0, totalTime, binDt)
    counts, _ = np.histogram(arrivalTimes, bins=edges)
    timeBins = edges[:-1] + binDt/2
    return arrivalTimes, counts, timeBins, Rp_i_out, (sigma_b, dist_type, alpha)

@njit
def simulation_loop_jit(arrivalTimes, pos, Rp_i, w0, w_z, Nres, nSteps, stepsPerSweep, dt, sigma, vFlow, Lres, Lbox,
                       includeBg, bgRate, beamShape, poisson_bg, perm_indices_x, perm_indices_y, perm_indices_z):
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
                if idx >= len(arrivalTimes):
                    print("underallocated mem")
                
                arrivalTimes[idx] = t0 + np.random.random() * dt
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

    if idx > len(arrivalTimes):
        print("Overflow: attempted", idx, ", allocated", len(arrivalTimes))
    arrivalTimes = arrivalTimes[:min(idx, len(arrivalTimes))]  # Trim safely
    
    return idx, Rp_i

def run_one_sim(sim_num):
    #fixed variables
    D = 1e-11
    dt = 5e-6
    vF = 5e-4
    tt = 60
    setDt = 500e-6
    bgRate = 1e2
    #sim specific ops
    seed = sim_num + int(time.time()) % 10000
    rng = np.random.default_rng(seed)
    start_sim = time.time()
    sim_dir = os.path.join(data_dir, f"sim_{sim_num:07}")
    os.makedirs(sim_dir, exist_ok=True)

    #sim ground truth
    num_species = np.random.randint(1,4) #[1,3]
    AmpS1 = rng.integers(50, 1900)
    AmpS2 = rng.integers(AmpS1+400, AmpS1+400+1350)
    AmpS3 = rng.integers(AmpS2+400, 5250)
    total_conc = rng.uniform(9e-12, 7e-11)
    
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
    sigma_b1 = 0
    sigma_b2 = None
    sigma_b3 = None


    #run sims
    arrivalTimes1, _, _, Rp_i1, params = SimPhotDiffFlowGL6(C_molar = conc1,
                                                    Rp = AmpS1 / setDt, 
                                                    D = D, 
                                                    totalTime = tt, 
                                                    binDt = dt, 
                                                    w0 = w0, 
                                                    axialFactor = axialFactor, 
                                                    includeBg = True, 
                                                    bgRate = bgRate, 
                                                    beamShape = 'gl',
                                                    vFlow=vF,
                                                    rng = rng,
                                                    num_species=num_species)
    truedist1 = Rp_i1
    at1 = arrivalTimes1
    sigma_b_1, dist_type, alpha = params
    sigma_b1 = sigma_b_1

    if num_species >= 2:
        arrivalTimes2, _, _, Rp_i2, params = SimPhotDiffFlowGL6(C_molar = conc2,
                                                        Rp = AmpS2 / setDt, 
                                                        D = D, 
                                                        totalTime = tt, 
                                                        binDt = dt, 
                                                        w0 = w0, 
                                                        axialFactor = axialFactor, 
                                                        includeBg = False, 
                                                        bgRate = 0, 
                                                        beamShape = 'gl',
                                                        vFlow=vF,
                                                        rng = rng,
                                                        num_species=num_species)
        truedist2 = Rp_i2
        at2 = arrivalTimes2
        sigma_b_2, _, _ = params
        sigma_b2 = sigma_b_2
        
    if num_species == 3:
        arrivalTimes3, _, _, Rp_i3, params = SimPhotDiffFlowGL6(C_molar = conc3,
                                                        Rp = AmpS3 / setDt, 
                                                        D = D, 
                                                        totalTime = tt, 
                                                        binDt = dt, 
                                                        w0 = w0, 
                                                        axialFactor = axialFactor, 
                                                        includeBg = False, 
                                                        bgRate = 0, 
                                                        beamShape = 'gl',
                                                        vFlow=vF,
                                                        rng = rng,
                                                        num_species=num_species)
        truedist3 = Rp_i3
        at3 = arrivalTimes3
        sigma_b_3, _, _ = params
        sigma_b3 = sigma_b_3

    fullBrightDist = np.concatenate([truedist1.flatten()*setDt, truedist2.flatten()*setDt, truedist3.flatten()*setDt])
    bins = np.linspace(0, 8000, 101)
    truedist, bin_edges_true_dist = np.histogram(fullBrightDist, bins = bins)
    # plt.bar(bin_edges_true_dist[:-1], truedist, width=np.diff(bin_edges_true_dist), edgecolor='black', align='edge')
    # plt.title('Underlying Brightness Distribution', fontsize=14)
    # plt.xlabel('Fluorescence Intensity', fontsize=12)
    # plt.ylabel('Number of particles', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)

    fullTOAs = np.concatenate([at1, at2, at3])
    fullTOAs = np.sort(fullTOAs)
    #{datetime.datetime.now().strftime("%m_%d_%H_%M")}
    bins_hist = np.linspace(0, tt, int((tt)/(setDt)) + 1)
    histA, _ = np.histogram(fullTOAs, bins_hist)
    PCHedges = np.logspace(np.log10(3), np.log10(8000), 51)
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
            "vFlow":vF,
            "bgRate":bgRate,
            "disttype":dist_type,
            "sigma_bS1":sigma_b1,
            "sigma_bS2":sigma_b2,
            "sigma_bS3":sigma_b3,
            "alpha":alpha
        },
        "Data":{
            "true_bins":truedist.tolist(),
            "pch_bins":PCHbins.tolist(),
            "true_edges":bin_edges_true_dist.tolist(),
            "pch_edges":PCHedges.tolist()
        },
        "Other":{
            "Seed":seed
        }
    }

    with open(os.path.join(sim_dir, "GT.json"), "w") as f:
        json.dump(GT, f)

    # np.save(os.path.join(sim_dir, "true_bins.npy"), truedist)
    # np.save(os.path.join(sim_dir, "true_edges.npy"), bin_edges_true_dist)
    # np.save(os.path.join(sim_dir, "pchbins.npy"), PCHbins)
    # np.save(os.path.join(sim_dir, "pchedges.npy"), PCHedges)
    
    #np.savez_compressed(os.path.join(sim_dir, "TOAs.npy"), TOAs = fullTOAs)
    print(f"Finished sim {sim_num}, Time: {time.time() - start_sim}")
    gc.collect()
    return 


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    num_sims = 1000000
    num_workers = 200

    with mp.Pool(processes=num_workers) as pool:
        pool.map(run_one_sim, range(num_sims))

