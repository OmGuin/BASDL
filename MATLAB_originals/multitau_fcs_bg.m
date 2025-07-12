function [tau, g2] = multitau_fcs_bg(I, dt, M, P, bgRate)
% multitau_fcs_bg  Compute and plot background-corrected FCS correlation
%
%   [tau, g2] = multitau_fcs_bg(I, dt, M, P, bgRate)
%
%   Inputs:
%     I       – Intensity trace (counts per bin)
%     dt      – Time resolution (bin width) [s]
%     M, P    – Multi-tau parameters (# channels per cascade, # cascade levels)
%     bgRate  – Background count rate [counts/s]
%
%   Outputs:
%     tau     – Lag times [s]
%     g2      – Background-corrected normalized autocorrelation g^{(2)}(τ)
%
%   This function calls the original multitau_fcs to get the raw normalized
%   correlation g2raw, applies Poisson background correction, and plots the
%   corrected g2 curve with a subtitle indicating background correction.

  % 1) Obtain raw autocorrelation
  [tau, g2raw] = multitau_fcs(I, dt, M, P);

  % 2) Calculate background fraction
  meanI    = mean(I);            % mean total counts per bin
  bgCounts = bgRate * dt;        % average background counts per bin
  f_b      = bgCounts / meanI;   % background fraction

  % 3) Correct the correlation amplitude
  g2 = (g2raw - 1) ./ (1 - f_b)^2 + 1;

  % % 4) Plot the background-corrected correlation
  % figure;
  % semilogx(tau, g2, 'o-','MarkerSize',6);
  % xlabel('\tau (s)');
  % ylabel('g^{(2)}(\tau)');
  % title('FCS Autocorrelation');
  % subtitle('Background-corrected');



% %% ——— Fit to 3D‐diffusion model ———
% % 1) Define the model
% ft = fittype(@(N, tauD, S, x) ...
%      1 + (1./N) .* 1./(1 + x./tauD) .* 1./sqrt(1 + x./(S^2 * tauD)), ...
%      'independent','x','coefficients',{'N','tauD','S'});
% 
% % 2) Fit options
% opts = fitoptions(ft);
% %opts.Method     = 'NonlinearLeastSquares';
% opts.StartPoint = [mean(g2)-1, median(tau), 5];  % [N_guess, tauD_guess, S_guess]
% opts.Lower      = [  0,            0,        1];  % N≥0, τD≥0, S≥1
% opts.Upper      = [ Inf,          Inf,      Inf];
% 
% % 3) Perform the fit
% [cfun, gof] = fit(tau(:), g2(:), ft, opts);
% 
% % 4) Overlay the fit on your semilogx plot
% hold on;
% semilogx(tau, cfun(tau), 'r-', 'LineWidth', 1.5);
% legend('Data','3D Diffusion Fit','Location','best');
% 
% % 5) Annotate fitted parameters
% txt = sprintf('N = %.2f\nτ_D = %.3g s\nS = %.2f\nR^2 = %.3f', ...
%                cfun.N, cfun.tauD, cfun.S, gof.rsquare);
% xpos = tau(round(end/5)); 
% ypos = min(g2) + 0.7*(max(g2)-min(g2));
% text(xpos, ypos, txt, 'FontSize',10, 'BackgroundColor','w', 'EdgeColor','k');


end
