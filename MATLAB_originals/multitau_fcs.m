function [tau, g2] = multitau_fcs(I, dt, M, P)
% MULTITAU_FCS  Compute FCS autocorrelation via the multi-τ algorithm
%
%  takes as input your fluorescence intensity trace I (a 1-D vector) and 
%  the basic sampling interval dt, and returns the normalized 
%   autocorrelation g2 and lag times tau, then plots the result.
%
%%%% Notes on tweaking
    % 
    % M sets how many points you sample per level (common choice: 16–32).
    % 
    % P is usually 2 (doubling the bin width each level).
    % 
    % If your data are photon‐arrival events rather than binned intensity, 
    % you should first bin into a regular time trace.
    % 
    % For very long records you may need to increase Lmax manually or 
    % truncate the input to balance speed vs. maximum lag.
%%%%%%%%%%%%%%%%%%%%
%
%   [tau, g2] = multitau_fcs(I, dt, M, P)
%
% Inputs:
%   I   – fluorescence time series (row or column vector)
%   dt  – sampling interval (seconds)
%   M   – number of channels per level (e.g. 16 or 32)
%   P   – coarse‐graining factor between levels (typically 2)
%
% Outputs:
%   tau – vector of lag times (seconds)
%   g2  – normalized autocorrelation g2(tau)
%
% Example:
%   dt = 1e-6;                  % 1 µs binning
%   I  = your_fluo_trace;      % e.g. [photon counts]
%   [tau, g2] = multitau_fcs(I, dt, 16, 2);
%   semilogx(tau, g2, '.', 'MarkerSize',12);
%   xlabel('Lag \tau (s)'); ylabel('g^{(2)}(\tau)');
%

%% ensure column vector
I = I(:);
N = numel(I);

%% determine number of levels
Lmax = floor(log((N-1)/(M-1)) / log(P)) + 1;

%% pre-allocate
G   = nan(Lmax, M);    % raw correlations
nrm = nan(Lmax, M);    % normalization counts

x_cur = I;             % current level data
meanI = mean(I);

%% loop over levels
for L = 1:Lmax
    Nl = numel(x_cur);
    % compute correlations at this level
    for m = 1:M
        if m < Nl
            prod = x_cur(1:end-m) .* x_cur(m+1:end);
            G(L,m)   = mean(prod);
            nrm(L,m) = 1;    % all time‐points counted equally in mean()
        else
            G(L,m)   = NaN;
            nrm(L,m) = NaN;
        end
    end
    
    % coarse‐grain for next level (except last)
    if L < Lmax
        % group in blocks of P (here P=2 by default)
        n_new = floor(Nl/P);
        x_next = zeros(n_new,1);
        for k = 1:n_new
            idx = (k-1)*P + (1:P);
            x_next(k) = mean(x_cur(idx));
        end
        x_cur = x_next;
    end
end

%% stitch together lag times and G
tau = [];
g2  = [];
for L = 1:Lmax
    % lags at this level (in units of dt):
    %   m = 0:(M-1), but skip the zero‐lag at higher levels to avoid duplicate
    if L == 1
        ms = 0:(M-1);
    else
        ms = 1:(M-1);
    end
    % convert to time:  tau = dt * ( P^(L-1) * ms )
    tau_L = dt * (P^(L-1)) * ms;
    
    % grab corresponding G values
    G_L = G(L, ms+1);
    
    % normalize:  g2 = <I(0)I(τ)> / <I>^2
    g2_L = G_L / (meanI^2);
    
    % accumulate
    tau = [tau, tau_L];
    g2  = [g2,  g2_L];
end


% % --- plot ---
% figure;
% semilogx(tau, g2, 'o-','MarkerSize',6,'LineWidth',1);
% grid on;
% xlabel('Lag time \tau (s)','FontSize',12);
% ylabel('g^{(2)}(\tau)','FontSize',12);
% title('FCS Autocorrelation (multi-\tau)','FontSize',14);

end
