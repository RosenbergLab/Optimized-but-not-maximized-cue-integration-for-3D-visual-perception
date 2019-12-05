%{
This code implements the neural network simulations presented in the paper
"Optimized but Not Maximized Cue Integration for 3D Visual Perception"
by Ting-Yu Chang, Lowell Thompson, Raymond Doudlah, Byounghoon Kim, Adhira Sunkara, and Ari Rosenberg
eNeuro, X(X): eXXXX, YEAR
%}

clear all;
close all hidden;
clc;

nNeurons = 72; % Number of neurons in each layer --> 5 degree spacing in tilt preferences
tiltPrefs = deg2rad(linspace(0, 360-360/nNeurons, nNeurons)); % Tilt preference of each neuron

tilts_deg = [0:1:359]; % Tilts with 1 deg spacing
tilts_rad = deg2rad(tilts_deg);

test_tilt_index = 181; % Index of presented stimulus tilt (Tilt = 180)
bin_size = pi/180; % 1 degree bins

cFunc = @(x,xdata) x(1)*exp(-x(2)*xdata) + x(3); % Lambda scaling function that equates neuronal population gain and behavioral precision
xmulti{1} = [3.7024 2.3668 0.0013]; % Parameters for Monkey L
xmulti{2} = [2.0303 2.4983 0.0015]; % Parameters for Monkey F

% Load combined-cue behavioral precisions and tuning curve parameters
load('NeuralNet_Parameters.mat');

%% Generate 3 Model Predictions
PoissNoise = 0; % Simulation with Poisson noise (or noiseless)
for m = 1:length(Combined_Sigmas) % For each monkey
    for s = 1:size(Combined_Sigmas{m},1) % For each slant
        for d = 1:size(Combined_Sigmas{m},2) % For each distance
            
            % Tuning curve parameters for the current slant & distance (based on neuronal recordings in area CIP) 
            kappaVal = tuning_curve_kappas{m}(s,d); % Kappa
            LambdaVal = cFunc(xmulti{m},kappaVal); % Lambda scaling value
            
            % Create combined-cue representations
            % Model 1: Three independent populations model: Left Eye Perspective, Right Eye Perspective, and Stereoscopic Populations
            A_ThreePop = AMP{m}(s,d,1) + AMP{m}(s,d,2) + AMP{m}(s,d,3);
            % Model 2: Two independent populations model: Perspective and Stereoscopic Populations
            A_TwoPop = (((AMP{m}(s,d,1).^2) + (AMP{m}(s,d,2).^2))./(AMP{m}(s,d,1) + AMP{m}(s,d,2))) + AMP{m}(s,d,3);
            % Model 3: One Population
            A_OnePop = ((AMP{m}(s,d,1).^2) + (AMP{m}(s,d,2).^2) + (AMP{m}(s,d,3).^2))./(AMP{m}(s,d,1) + AMP{m}(s,d,2) + AMP{m}(s,d,3));
            
            % Create likelihood functions given the tuning curves - with or without Poisson noise (parameter PoissNoise)
            ThreePop_PPC_likelihood{m}(s,d,:) = generate_likelihood(LambdaVal, A_ThreePop, tiltPrefs, kappaVal, test_tilt_index, tilts_rad, PoissNoise);
            TwoPop_PPC_likelihood{m}(s,d,:) = generate_likelihood(LambdaVal, A_TwoPop, tiltPrefs, kappaVal, test_tilt_index, tilts_rad, PoissNoise);
            OnePop_PPC_likelihood{m}(s,d,:) = generate_likelihood(LambdaVal, A_OnePop, tiltPrefs, kappaVal, test_tilt_index, tilts_rad, PoissNoise);
            
            % Create posteriors from the likelihoods
            ThreePop_PPC_posterior{m}(s,d,:) = posterior_from_likelihood(squeeze(ThreePop_PPC_likelihood{m}(s,d,:)),bin_size);
            TwoPop_PPC_posterior{m}(s,d,:) = posterior_from_likelihood(squeeze(TwoPop_PPC_likelihood{m}(s,d,:)),bin_size);
            OnePop_PPC_posterior{m}(s,d,:) = posterior_from_likelihood(squeeze(OnePop_PPC_likelihood{m}(s,d,:)),bin_size);
            
            % Fit a Gaussian probability density function to estimate precision
            [ThreePop_PPC_sigma{m}(s,d), ThreePop_PPC_mu{m}(s,d)] = fit_posterior_sigma(squeeze(ThreePop_PPC_posterior{m}(s,d,:)),tilts_rad);
            [TwoPop_PPC_sigma{m}(s,d), TwoPop_PPC_mu{m}(s,d)] = fit_posterior_sigma(squeeze(TwoPop_PPC_posterior{m}(s,d,:)),tilts_rad);
            [OnePop_PPC_sigma{m}(s,d), OnePop_PPC_mu{m}(s,d)] = fit_posterior_sigma(squeeze(OnePop_PPC_posterior{m}(s,d,:)),tilts_rad);
        end
    end
end

%% Plot decoded vs observed precisions
model_colors = [241 90 36; 0 255 0; 255 0 255]./255;
figure; set(gcf,'color','w'); hold on;
title('Model PPC Cue Combination');
p(1) = plot(1./Combined_Sigmas{1}(:).^2, 1./ThreePop_PPC_sigma{1}(:).^2, 'o', 'Color',model_colors(1,:),'MarkerFaceColor',model_colors(1,:)); % Plot data for Monkey L
plot(1./Combined_Sigmas{2}(:).^2, 1./ThreePop_PPC_sigma{2}(:).^2, 'o','Color',model_colors(1,:),'MarkerFaceColor',model_colors(1,:)); % Plot data for Monkey F

p(2) = plot(1./Combined_Sigmas{1}(:).^2, 1./TwoPop_PPC_sigma{1}(:).^2, 'o','Color',model_colors(2,:),'MarkerFaceColor',model_colors(2,:));
plot(1./Combined_Sigmas{2}(:).^2, 1./TwoPop_PPC_sigma{2}(:).^2, 'o','Color',model_colors(2,:),'MarkerFaceColor',model_colors(2,:));

p(3) = plot(1./Combined_Sigmas{1}(:).^2, 1./OnePop_PPC_sigma{1}(:).^2, 'o','Color',model_colors(3,:),'MarkerFaceColor',model_colors(3,:));
plot(1./Combined_Sigmas{2}(:).^2, 1./OnePop_PPC_sigma{2}(:).^2, 'o','Color',model_colors(3,:),'MarkerFaceColor',model_colors(3,:));

lim = axis;

% Identity and limits
r = refline(1,0); r.LineStyle = '--'; r.Color = 'k';
axis([min(lim),max(lim),min(lim),max(lim)]);
textSize = 14;
xlabel('Precision (observed)');
ylabel('Precision (decoded)');
legend(p(:),{'3 Populations','2 Populations', '1 Population'});
set(findall(gca, 'Type','text'), 'FontSize', textSize);
axis square; box on;