function InteractiveFCSArrivalPlot_flow(arrivalTimes, w0, S_fixed, conc, amp)
% InteractiveFCSArrivalPlot_flow  GUI to run FCS with flow model on photon arrivals
%   InteractiveFCSArrivalPlot_flow(arrivalTimes, w0, S_fixed)
%     arrivalTimes – sorted photon arrival timestamps [s]
%     w0           – lateral beam waist radius [m]
%     S_fixed      – (optional) structure parameter S to hold fixed in the fit
    
    arrivalTimes = arrivalTimes(:);
    if nargin < 3
        S_fixed = [];
    end

    % Default UI parameters
    defaultBinDt = (arrivalTimes(end) - arrivalTimes(1))/2e6;
    defaultM     = 64;
    defaultP     = 2;
    defaultBg    = 0;

    % Create figure
    hFig = figure('Name','FCS Flow Explorer','NumberTitle','off', ...
                  'MenuBar','none','ToolBar','none', ...
                  'Units','normalized','Position',[0.2 0.2 0.6 0.6]);

    % Axes for correlation plot
    hAx = axes('Parent',hFig,'Units','normalized', ...
               'Position',[0.1 0.4 0.85 0.55]);

    % UI controls
    labelW = 0.12; editW = 0.1; spacing = 0.02; y0 = 0.25;
    uicontrol('Style','text','Parent',hFig,'Units','normalized', ...
        'Position',[0.1 y0 labelW 0.05],'String','BinDt (s):','HorizontalAlignment','right');
    hBin = uicontrol('Style','edit','Parent',hFig,'Units','normalized', ...
        'Position',[0.1+labelW+spacing-0.02 y0 editW 0.05],'String',num2str(defaultBinDt));

    uicontrol('Style','text','Parent',hFig,'Units','normalized', ...
        'Position',[0.3 y0 labelW 0.05],'String','M channels:','HorizontalAlignment','right');
    hM = uicontrol('Style','edit','Parent',hFig,'Units','normalized', ...
        'Position',[0.3+labelW+spacing y0 editW 0.05],'String',num2str(defaultM));

    uicontrol('Style','text','Parent',hFig,'Units','normalized', ...
        'Position',[0.5 y0 labelW 0.05],'String','P factor:','HorizontalAlignment','right');
    hP = uicontrol('Style','edit','Parent',hFig,'Units','normalized', ...
        'Position',[0.5+labelW+spacing y0 editW 0.05],'String',num2str(defaultP));

    uicontrol('Style','text','Parent',hFig,'Units','normalized', ...
        'Position',[0.7 y0 labelW 0.05],'String','bgRate (Hz):','HorizontalAlignment','right');
    hBg = uicontrol('Style','edit','Parent',hFig,'Units','normalized', ...
        'Position',[0.7+labelW+spacing y0 editW 0.05],'String',num2str(defaultBg));

    hFitChk = uicontrol('Style','checkbox','Parent',hFig,'Units','normalized', ...
        'Position',[0.1 y0-0.08 0.2 0.05],'String','Fit Flow Model','FontSize',10);

    hRun = uicontrol('Style','pushbutton','Parent',hFig,'Units','normalized', ...
        'Position',[0.5 y0-0.1 0.1 0.07],'String','Run FCS','FontSize',10,'Callback',@runFCS);
    %uicontrol('Style','pushbutton','Parent',hFig,'Units','normalized', ...
    %    'Position',[0.85 y0-0.1 0.1 0.07],'String','Exit','FontSize',10,'Callback',@(~,~) close(hFig));
    uicontrol('Style','pushbutton','Parent',hFig,'Units','normalized', ...
    'Position',[0.85 y0-0.1 0.1 0.07],'String','Exit','FontSize',10,'Callback',@saveAndClose);


    % Cache for correlation
    persistent lastBinDt lastM lastP lastBgRate lastTau lastG2;

    % Initial run
    runFCS();

    function runFCS(~,~)
        set(hRun,'Enable','off'); drawnow;
        try
            % Read & validate inputs
            binDt  = str2double(get(hBin,'String'));
            M      = round(str2double(get(hM,'String')));
            P      = round(str2double(get(hP,'String')));
            bgRate = str2double(get(hBg,'String'));
            doFit  = get(hFitChk,'Value');

            if any([isnan(binDt)||binDt<=0, isnan(M)||M<1, isnan(P)||P<1, isnan(bgRate)||bgRate<0])
                error('Invalid parameters');
            end

            % Compute or reuse correlation
            if isempty(lastBinDt) || binDt~=lastBinDt || M~=lastM || ...
               P~=lastP || bgRate~=lastBgRate
                [tau, g2] = multitau_fcs_arrival(arrivalTimes, binDt, M, P, bgRate);
                lastBinDt  = binDt;
                lastM      = M;
                lastP      = P;
                lastBgRate = bgRate;
                lastTau    = tau;
                lastG2     = g2;
            else
                tau = lastTau;
                g2  = lastG2;
            end

            % Plot data
            cla(hAx);
            semilogx(hAx, tau, g2, '.','MarkerSize',8);
            hold(hAx,'on');
            xlabel(hAx,'\tau (s)');
            ylabel(hAx,'g^{(2)}(\tau)');
            grid(hAx,'on');
            title(hAx,'FCS Autocorrelation');
            delete(findall(hAx,'Tag','FitParamsBox'));
            legend(hAx,'off');

            % Fit flow model
            if doFit
                if isempty(S_fixed)
                    % Fit A, tauD, S, v
                    ft = fittype(@(A,tauD,S,v,x) ...
                        1 + A./(1 + x./tauD)./sqrt(1 + x./(S^2.*tauD)) .* ...
                        exp(-v^2.*x.^2./(w0^2.*(1 + x./tauD))), ...
                        'independent','x','coefficients',{'A','tauD','S','v'});
                    opts = fitoptions(ft);
                    opts.StartPoint = [max(g2)-1, median(tau), 5, 0];
                    opts.Lower      = [0,          0,         1, 0];
                    opts.Upper      = [Inf,        Inf,    Inf, Inf];
                else
                    % Fit A, tauD, v with S fixed
                    ft = fittype(@(A,tauD,v,x) ...
                        1 + A./(1 + x./tauD)./sqrt(1 + x./(S_fixed^2.*tauD)) .* ...
                        exp(-v^2.*x.^2./(w0^2.*(1 + x./tauD))), ...
                        'independent','x','coefficients',{'A','tauD','v'});
                    opts = fitoptions(ft);
                    opts.StartPoint = [max(g2)-1, median(tau), 0];
                    opts.Lower      = [0,          0,         0];
                    opts.Upper      = [Inf,        Inf,       Inf];
                end

                % Perform fit
                [cfun, gof] = fit(tau(:), g2(:), ft, opts);

                % Extract parameters
                A_fit    = cfun.A;
                tauD_fit = cfun.tauD;
                if isempty(S_fixed)
                    S_fit = cfun.S;
                else
                    S_fit = S_fixed;
                end
                v_fit = cfun.v;
                N_fit = 1/A_fit;
                D_fit = w0^2/(4*tauD_fit);

                % Overlay fit
                semilogx(hAx, tau, cfun(tau), 'rx','LineWidth',0.5);
                legend(hAx,{'Data','Flow Fit'},'Location','best');

                % Annotation text
                if isempty(S_fixed)
                    txt = sprintf(...
                        'N=%.2f\n\\tau_D=%.3g s\nS=%.2f\nv=%.3g m/s\nD=%.3g m^2/s\nR^2=%.3f', ...
                        N_fit, tauD_fit, S_fit, v_fit, D_fit, gof.rsquare);
                else
                    txt = sprintf(...
                        'N=%.2f\n\\tau_D=%.3g s\nv=%.3g m/s\nD=%.3g m^2/s\nR^2=%.3f', ...
                        N_fit, tauD_fit, v_fit, D_fit, gof.rsquare);
                end

                % Place annotation
                xl = get(hAx,'XLim'); yl = get(hAx,'YLim');
                xPos = 10^(log10(xl(1)) + 0.8*(log10(xl(2))-log10(xl(1))));
                yPos = yl(1) + 0.5*(yl(2)-yl(1));
                text(hAx, xPos, yPos, txt, ...
                     'Units','data','BackgroundColor','white', ...
                     'EdgeColor','black','Tag','FitParamsBox','FontSize',14);
            end

        catch ME
            errordlg(ME.message,'Error during FCS');
        end
        set(hRun,'Enable','on');
    end
    function saveAndClose(~,~)
        % Create output folder if it doesn't exist
        folderPath = fullfile(pwd, 'dataset_testing');
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
        end

        % Format filename using conc and amp
        concStr = strrep(num2str(conc, '%.1e'), '-', 'm');  % '7e-11' → '7m11'
        ampStr = num2str(amp);
        filenameBase = sprintf('FCS_Curve_%s_%s', concStr, ampStr);
        filenamePNG = fullfile(folderPath, [filenameBase '.png']);
        filenameFIG = fullfile(folderPath, [filenameBase '.fig']);

        % Get parameter values from GUI
        binDt  = str2double(get(hBin, 'String'));
        M      = round(str2double(get(hM, 'String')));
        P      = round(str2double(get(hP, 'String')));
        bgRate = str2double(get(hBg, 'String'));

        % Create export figure and copy axes
        hExportFig = figure('Visible', 'off');
        hExportAx = copyobj(hAx, hExportFig);
        set(hExportAx, 'Units', 'normalized', 'Position', [0.13 0.35 0.775 0.6]);

        % Build horizontal parameter string
        paramText = sprintf('BinDt = %.3g s   |   M = %d   |   P = %d   |   bgRate = %.1f Hz', ...
                            binDt, M, P, bgRate);

        % Add annotation below the plot (no border)
        annotation(hExportFig, 'textbox', [0.1, 0.15, 0.8, 0.05], ...
            'String', paramText, ...
            'FontSize', 12, ...
            'FontWeight', 'bold', ...
            'EdgeColor', 'none', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Interpreter', 'tex');

        % Save to PNG and .fig
        saveas(hExportFig, filenamePNG);
        savefig(hExportFig, filenameFIG);

        % Close both figures
        close(hExportFig);
        close(hFig);
    end



end

    