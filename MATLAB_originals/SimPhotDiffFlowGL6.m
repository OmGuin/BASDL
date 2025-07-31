function [arrivalTimes, counts, timeBins] = SimPhotDiffFlowGL6(C_molar, Rp, D, totalTime, binDt, w0, axialFactor, includeBg, bgRate, beamShape, vFlow, resFactor)
% SimPhotDiffFlowGL6  Simulate photon arrivals via sweeping reservoir + reflection
% Core elements:
%   • Pre-generated reservoir spanning resFactor×Lbox in x, ±Lbox in y,z
%   • Advective flow + diffusion each dt-step
%   • Diffusion: reflecting boundaries in excitation volume (±Lbox)
%   • No births/deaths: sweeping reservoir reused via column-wise permutation
%   • Photon thinning via fixed dt binning
%   • Binning and preview via RebinIntensity
%     This two-stage randomness
% 
%     Stage 1 (brightness jitter): Ri​=bi​Rp​ with bi​ drawn once from a non-Poisson distribution.
% 
%     Stage 2 (shot noise): photon counts in each Δt are Poisson with mean ∑​Ri​Wi​Δt.
% %
% Inputs:
%   C_molar     – concentration [mol/L]
%   Rp          – emission rate at beam center [photons/s]
%   D           – diffusion coefficient [m^2/s]
%   totalTime   – total simulation duration [s]
%   binDt       – time bin width [s]
%   w0          – lateral beam waist [m]
%   axialFactor – ratio w_z/w0 (unitless)
%   includeBg   – boolean: include Poisson background
%   bgRate      – background rate [photons/s]
%   beamShape   – 'gaussian' or 'gl'
%   vFlow       – advective flow speed in x [m/s]
%   resFactor   – reservoir length factor (multiples of Lbox) in x (default 10)
% Outputs:
%   arrivalTimes – photon timestamps [s]
%   counts       – counts per bin
%   timeBins     – bin centers [s]

tic
    % Defaults
    if nargin<12, resFactor = 10; end
    if nargin<11, vFlow = 0; end

    % 1) Geometry & constants
    NA   = 6.022e23;                   % Avogadro (#/mol)
    C_m3 = C_molar * 1e3;              % mol/L → mol/m^3
    w_z  = axialFactor * w0;           % axial waist [m]
    Lbox = 5 * max(w0, w_z);           % half-box size [m]
    Lres = resFactor * Lbox;           % reservoir half-length [m]

    % 2) Reservoir initialization
    area_yz = (2*Lbox)^2;              % cross-section area [m^2]
    Vres = (2*Lres) * area_yz;         % reservoir volume [m^3]
    Nres = max(1, poissrnd(C_m3 * NA * Vres)); % reservoir molecules (>=1)
    % Uniform positions in x∈[-Lres,Lres], y,z∈[-Lbox,Lbox]
    pos = [ (rand(Nres,1)-0.5)*2*Lres, (rand(Nres,2)-0.5)*2*Lbox ];

    % 3) Time step and diffusion
    dt    = binDt;                     % timestep [s]
    sigma = sqrt(2 * D * dt);          % diffusion std [m]
    nSteps = ceil(totalTime / dt);
    if vFlow>0
        stepsPerSweep = ceil((2*Lres)/(vFlow*dt));
    else
        stepsPerSweep = inf;
    end

    % 4) Preallocate photon times
    Veff = pi^(3/2) * w0^2 * w_z;      % focal vol [m^3]
    Navg = C_m3 * NA * Veff;           % avg in focus
    expCount = ceil((Navg*Rp + bgRate) * totalTime * 1.2);
    arrivalTimes = zeros(expCount,1);
    idx = 1;

    % 4.5) Other noise effects in emission (particle brightness variation)
    % std of brightness fluctuations (e.g. 0.2 for ±20% jitter)
    sigma_b = 0;
    % per-molecule brightness factors, mean = 1
    b = exp(sigma_b * randn(Nres,1) - 0.5*sigma_b^2);

    % 5) Simulation loop
    for k = 1:nSteps
        t0 = (k-1)*dt;

        % 5a) Advect reservoir in x
        pos(:,1) = pos(:,1) + vFlow * dt;
        % 5a-ii) Periodic wrap full reservoir in x
        pos(:,1) = mod(pos(:,1) + Lres, 2*Lres) - Lres;

        % 5b) Diffuse particles within excitation volume
        inBox = abs(pos(:,1)) <= Lbox;
        pos(inBox,:) = pos(inBox,:) + sigma * randn(sum(inBox),3);

        % 5c) Reflecting boundaries for diffusion inBox
        % reflect x
        ix = inBox & pos(:,1) > +Lbox;
        pos(ix,1) = 2*Lbox - pos(ix,1);
        ix = inBox & pos(:,1) < -Lbox;
        pos(ix,1) = -2*Lbox - pos(ix,1);
        % reflect y
        iy = inBox & pos(:,2) > +Lbox;
        pos(iy,2) = 2*Lbox - pos(iy,2);
        iy = inBox & pos(:,2) < -Lbox;
        pos(iy,2) = -2*Lbox - pos(iy,2);
        % reflect z
        iz = inBox & pos(:,3) > +Lbox;
        pos(iz,3) = 2*Lbox - pos(iz,3);
        iz = inBox & pos(:,3) < -Lbox;
        pos(iz,3) = -2*Lbox - pos(iz,3);

        % 5d) Photon emission + background counts
        xy2 = pos(:,1).^2 + pos(:,2).^2;
        z2  = pos(:,3).^2;
        switch lower(beamShape)
            case 'gaussian'
                W = exp(-2*xy2/w0^2 - 2*z2/w_z^2);
            case 'gl'
                Wlat = exp(-2*xy2/w0^2);
                Wax  = 1 ./ (1 + z2./w_z.^2);
                W    = Wlat .* Wax;
            otherwise
                error('Unknown beamShape: %s', beamShape);
        end
%        Rtot = Rp * sum(W);
        Rtot = Rp * sum( b .* W );     % photons/s with particle brightness variation
        Nph  = poissrnd(Rtot * dt);   %Shot noise of quantum process
        if includeBg
            Nbg = poissrnd(bgRate * dt);
        else
            Nbg = 0;
        end
        NtotEv = Nph + Nbg;
        if NtotEv > 0
            arrivalTimes(idx:idx+NtotEv-1) = t0 + rand(NtotEv,1)*dt;
            idx = idx + NtotEv;
        end

        % 5e) Permute reservoir columns every full sweep
        if mod(k, stepsPerSweep) == 0
            perm = randperm(Nres);
            pos(:,1) = pos(perm,1);
            perm = randperm(Nres);
            pos(:,2) = pos(perm,2);
            perm = randperm(Nres);
            pos(:,3) = pos(perm,3);
        end
    end

    % 6) Trim arrivals
    arrivalTimes = arrivalTimes(1:idx-1);

    % 7) Bin into intensity trace
    edges    = 0:binDt:totalTime;
    counts   = histcounts(arrivalTimes, edges);
    timeBins = edges(1:end-1) + binDt/2;

    % 8) Preview plot (coarse rebin)
    newDt = min(binDt*100, totalTime/1000);
%    newDt = 0.0005;
    [newTimeBins, newCounts] = RebinIntensity(timeBins, counts, newDt);
    

    toc

end

function [newTimeBins, newCounts] = RebinIntensity(timeBins, counts, newDt)
% RebinIntensity  Efficiently re-bin an intensity trace to a new time resolution
%
%   [newTimeBins, newCounts] = RebinIntensity(timeBins, counts, newDt)
%
%   Inputs:
%     timeBins    – Original bin-center times [s]
%     counts      – Photon counts per original bin
%     newDt       – Desired new bin width [s]
%
%   Outputs:
%     newTimeBins – New bin-center times [s]
%     newCounts   – Photon counts per new bin

% 1) Original bin width and span
tOld = timeBins(2) - timeBins(1);
tStart = timeBins(1) - tOld/2;
tEnd   = timeBins(end) + tOld/2;

% 2) Number of new bins
duration = tEnd - tStart;
nNew = floor(duration / newDt);

% 3) Compute new bin indices for each original bin
defIdx = floor((timeBins - tStart) / newDt) + 1;
% ensure indices are within [1, nNew]
binIdx = min(max(defIdx, 1), nNew);

% 4) Sum counts into new bins (vectorized)
newCounts = accumarray(binIdx(:), counts(:), [nNew, 1]);

% 5) Compute new bin centers
newTimeBins = tStart + ( (0:(nNew-1))'*newDt ) + newDt/2;

% 6) Plot the rebinned intensity trace
figure;
plot(newTimeBins, newCounts, '-');
xlabel('Time (s)');
ylabel('Counts per bin');
title('Re-binned Intensity Trace');
grid on;

end
