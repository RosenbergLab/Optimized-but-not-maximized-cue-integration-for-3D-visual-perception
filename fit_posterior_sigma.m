function [posterior_sigma,posterior_mu] = fit_posterior_sigma(posterior,tilts_rad)
    %% Fit a Gaussian to the posterior distribution
    % Inputs:
    % posterior: the full posterior distribution
    % tilts_rad: tilts in radians used to fit the pdf
    
    gaussian = @(p,x) normpdf(x,p(1),p(2)); % Define a normal pdf
    opts = optimset('Display','off'); % Don't show output
    min_sig = 1/sqrt(18); % Max precision that can be estimated given the tilt sampling interval used in the study
    try
        out = lsqcurvefit(gaussian,[pi,1],tilts_rad,posterior',[0,min_sig],[2*pi,200],opts); % Fit gaussian
    catch e
        fprintf(1,'The identifier was:\n%s',e.identifier);
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        out = [NaN,NaN];
    end
    posterior_sigma = out(2);
    posterior_mu = out(1);
end