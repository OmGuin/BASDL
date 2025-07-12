function [tau, g2] = multitau_fcs_arrival(arrivalTimes, binDt, M, P, bgRate)
% multitau_fcs_arrival  FCS from photon timestamps with low memory footprint
%
%   [tau, g2] = multitau_fcs_arrival(arrivalTimes, binDt, M, P)
%
%   Inputs:
%     arrivalTimes – sorted photon arrival times [s]
%     binDt        – time‐bin width for histogramming [s]
%     M, P         – multi‐tau parameters (#channels, #levels)
%
%   Outputs:
%     tau, g2      – lag times and normalized correlation

  % 1) Decide on number of bins
  Tmax = arrivalTimes(end);                 
  Nb   = ceil(Tmax/binDt);     

  % 2) Map each timestamp to a bin index
  %    (timestamps == 0 go to bin 1;  
  %     timestamps right at a boundary go to the upper bin)
  binIdx = min( ceil(arrivalTimes/binDt), Nb );

  % 3) Build counts‐per‐bin via accumarray (no edges array!)
  counts = accumarray(binIdx, 1, [Nb,1]);

  % 4) Compute correlation on the binned trace
  [tau, g2] = multitau_fcs_bg(counts, binDt, M, P, bgRate);

end
