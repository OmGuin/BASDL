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