function posterior = posterior_from_likelihood(likelihood,bin_size)
    %% Scale the likelihood function to have unit area -- so we can fit a probability density function
    % Inputs:
    % likelihood: the likelihood function
    % bin_size: the spacing in radians between evaluated points of the likelihood
    
    % Normalize
    posterior = likelihood./sum(likelihood);
    posterior = posterior./bin_size;
end