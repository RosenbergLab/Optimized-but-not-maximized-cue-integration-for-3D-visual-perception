function [likelihood, out_r, TuningKernels] = generate_likelihood(LambdaVal, AmpVal, tiltPrefs, kappaVal, test_stim_index, tilts_rad, poiss)
%% Generate a likelihood function using von Mises tuning curves
% Inputs:
% LambdaVal: Scaling parameter to equate population gain and behavioral precision
% AmpVal: Max amplitude for the individual tuning curves
% tiltPrefs: A vector of the preferred tilt values for each neuron in radians
% kappaVal: Kappa value for the tuning curves
% test_stim_index: index of the stimulus tilt
% tilts_rad: tilts for building tuning curves
% poiss: simulate with poisson noise (or not)

tuning_fun = @(A,mu,x) A.*exp(-kappaVal)*exp(kappaVal*cos(x-mu)); % Inline Von mises function for generating tuning curves.

for i = 1:length(tiltPrefs) % Define N tuning curves.   
    temp_Tuning = tuning_fun(AmpVal, tiltPrefs(i), tilts_rad); %Generate tuning curve of the ith neuron.
    if poiss % Poisson noise?
        unscaled_rMatrix(i,:) = poissrnd(round(temp_Tuning)); % Unscaled response vector
        r_vec(i,:) = LambdaVal.*unscaled_rMatrix(i,:); % Response vector scaled by Lambda
    else
        unscaled_rMatrix(i,:) = temp_Tuning;
        r_vec(i,:) = LambdaVal.*temp_Tuning;
    end
    TuningCurveStack(i,:) = temp_Tuning;
    TuningKernels(i,:) = log(TuningCurveStack(i,:)); % Define kernels: log of the tuning curves
end
out_r = unscaled_rMatrix(:,test_stim_index);% Used for figures
if 0 ==  min(TuningCurveStack(:)) % Error check
    'log of zero taken'
end
% p(r|s) ~ exp(hT(s) * r); the kernels h(s) are identical across input layers (for each slant/distance), up to an additive constant.
likelihood = exp(TuningKernels' * r_vec(:,test_stim_index));
end