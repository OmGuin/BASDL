function ReadMe_PhotoArrival_and_FCS()
% ReadMe  Display usage instructions for SimPhotDiffFlowGL and InteractiveFCSArrivalPlot_flow
%
%   Call ReadMe in MATLAB to see step-by-step guidance.

    fprintf('\nUsing SimPhotDiffFlowGL and InteractiveFCSArrivalPlot_flow:\n\n');
    fprintf('1) Set up simulation parameters (run FCSparams.m  :\n');
    fprintf('   - C_molar: concentration (mol/L)\n');
    fprintf('   - Rp: photon emission rate at beam center (photons/s)\n');
    fprintf('   - D: diffusion coefficient (m^2/s)\n');
    fprintf('   - w0: lateral beam waist (m)\n');
    fprintf('   - axialFactor: ratio of axial-to-lateral waist\n');
    fprintf('   - totalTime: simulation duration (s)\n');
    fprintf('   - binDt: time bin width for intensity trace (s)\n');
    fprintf('   - includeBg: true/false for background counts\n');
    fprintf('   - bgRate: background count rate (Hz)\n');
    fprintf('   - beamShape: ''gaussian'' or ''gl'' (Gaussian-Lorentzian)\n');
    fprintf('   - vFlow: lateral flow speed (m/s)\n');
    fprintf('   - (optional) S_fixed: fix structure parameter in GUI fit\n\n');
    fprintf('2) Generate photon arrival data:\n');
    fprintf('   [arrivalTimes, counts, timeBins] = ...\n');
    fprintf('       SimPhotDiffFlowGL(C_molar, Rp, D, totalTime, binDt, w0, axialFactor, ...\n');
    fprintf('                            includeBg, bgRate, beamShape, vFlow);\n\n');
    fprintf('3) Launch FCS GUI:\n');
    fprintf('   InteractiveFCSArrivalPlot_flow(arrivalTimes, w0, S_fixed);\n\n');
    fprintf('4) In the GUI:\n');
    fprintf('   - Adjust BinDt, M channels, P factor, bgRate\n');
    fprintf('   - Check ''Fit Flow Model'' to enable fitting\n');
    fprintf('   - Click Run FCS (button will gray out during computation)\n');
    fprintf('   - View semilog-x plot of g^{(2)}(tau) with fit overlay\n');
    fprintf('   - Annotation box shows N, tau_D, S (if fitted), v, D, and R^2\n\n');
    fprintf('5) Iterate or Exit:\n');
    fprintf('   - Change inputs and click Run FCS again to update\n');
    fprintf('   - Click Exit to close the GUI\n\n');
    fprintf(' ');
    fprintf(' IN USE: \n\n');
    fprintf('FCSparams');
    fprintf('[arrivalTimes, counts, timeBins] = SimPhotDiffFlowGL6(C_molar, Rp, D, totalTime, binDt, w0, axialFactor, includeBg, bgRate, "gl",5e-3); \n\n ');
    fprintf('InteractiveFCSArrivalPlot_flow(arrivalTimesFb, w0, 3) \n');
end

