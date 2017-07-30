% =========================================================================
% =========================================================================
% These files are free for non-commercial use. If commercial use is
% intended, you can contact me at e-mail given below. The files are intended
% for research use by experienced Matlab users and includes no warranties
% or services.  Bug reports are gratefully received.
% I wrote these files while I was working as scientific researcher
% with University of Belgrade - Innovation Center at Faculty of Mechanical
% Engineering so as to test my research ideas and show them to wider
% community of researchers. None of the functions are meant to be optimal,
% although I tried to (some extent) speed up execution. You are wellcome to
% use these files for your own research provided you mention their origin.
% If you would like to contact me, my e-mail is given below.
% =========================================================================
% If you use this code for your own research, please cite it as:
% [1] Vukovic,N., Miljkovic,Z., Robust Sequential Learning of Feedforward
% Neural Networks in the Presence of Heavy-Tailed Noise, Neural Networks,
% Vol. 63, pp.31-47, Elsevier, 2015.
% [2] Vukovic,N., Mitic,M., Miljkovic,Z., Variational Inference for Robust
% Sequential Learning of Multilayered Perceptron Neural Network,
% FME Transactions Vol.43 No.2, pp. 123-130,
% University of Belgrade – Faculty of Mechanical Engineering, 2015.
% =========================================================================
% =========================================================================
% All rights reserved by: Dr. Najdan Vukovic
% contact e-mail: najdanvuk@gmail.com or nvukovic@mas.bg.ac.rs
% =========================================================================
% =========================================================================
clc,clear,close all
% Looking for optimal setting of parameters...
% =========================================================================
% Note: you need to add all folders to path (select folder, right click=> add folders...)
% =========================================================================
% =========================================================================
% =========================================================================
%==========================================================================
%                       T E S T   S E T 
% =========================================================================
% =========================================================================
%                 Mackey_Glass timeseries
% =========================================================================
Mackey_Glass_timeseries

% % add outliers
pct_of_outliers = 0.1 % 0-1 
number_of_outliers = round(pct_of_outliers * size(Ytrain,2))
offset_of_outliers = .1

index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers)

Youtlier  = Ytrain(:,index_of_outliers) + sqrt(offset_of_outliers) .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers))
Xoutlier = Xtrain(:,index_of_outliers)

plot([index_of_outliers ],Youtlier,'or');hold on

Ytrain(:,index_of_outliers) = Youtlier

plot(Ytrain)

% =========================================================================
% =========================================================================
%               I N I T I A L I Z E   P A R A M E T E R S
% =========================================================================
% =========================================================================
nf = 10                     % number of functions
xS0 = 1.5                   % spread
initVar = .10               % inital variance for weights initialization
bias = 'with_bias'
initBias = .10;             % inital bias for weights initialization
p0 = 9e1                    % inital state uncertainty
q0 = 1e-3                   % process covariance
r0 = 3e0                    % measurement covariance
ne = 1                      % pct of data to be used in training
DoF_h = 10
number_of_trials = 10       % number of trials
Emin = 1e-100              % for a record: just to let it work
% =========================================================================
[ rowY , colY ] = size(Ytrain)
[ rowX , colX ] = size(Xtrain)
%==========================================================================
%==========================================================================
% Prepare data for sequential training
% Randomize data for training
N = randomize_data(colY, round(ne * colY));
% Define training and testing set
Xtrain = Xtrain(:,N);Ytrain = Ytrain(:,N);
% Xtest = Xreal; Ytest = Yreal;

% normalize data
% Xn = [Xtrain Xtest];
% Yn = [Ytrain Ytest];
% Xtrain = (Xtrain - min(Xn(:))) ./ (max(Xn(:)) - min(Xn(:)))
% Ytrain = (Ytrain - min(Yn(:))) ./ (max(Yn(:)) - min(Yn(:)))
% Xtest = (Xtest - min(Xn(:))) ./ (max(Xn(:)) - min(Xn(:)))
% Ytest = (Ytest - min(Yn(:))) ./ (max(Yn(:)) - min(Yn(:)))
%==========================================================================
%==========================================================================

% EKF OR
hbf_or_ekf_parameters.number_of_functions = nf;
hbf_or_ekf_parameters.xS0 = xS0;
hbf_or_ekf_parameters.initVar = initVar;
hbf_or_ekf_parameters.initBias = initBias;
hbf_or_ekf_parameters.p0 = p0;            % inital state uncertainty
hbf_or_ekf_parameters.q0 = q0;            % process covariance
hbf_or_ekf_parameters.r0 = r0;           % measurement covariance
hbf_or_ekf_parameters.ne = ne;            % % pct of data to be used in training
hbf_or_ekf_parameters.DoF_h = DoF_h;
hbf_or_ekf_parameters.bias = bias;
hbf_or_ekf_parameters.Emin = Emin;
%==========================================================================
%==========================================================================
%                              T R A I N I N G
% =========================================================================
%==========================================================================

HBF_OR_EKF_Parameters = zeros( number_of_trials , 6 );
HBF_OR_EKF_Error = zeros( number_of_trials , ne * colY );
Y_HBF_EKF_OR = zeros(number_of_trials , size(Ytest,2));

for trial = 1 : number_of_trials
    
    %     EKF OR
    [ x_or_ekf , Er_or_ekf , Statistics_or_ekf ] = HBF_robust_EKF_training_Decembar_2013( Xtrain , Ytrain , Xtest , Ytest , hbf_or_ekf_parameters);
    HBF_OR_EKF_Parameters(trial,:) = Statistics_or_ekf;
    %     HBF_OR_EKF_Error(trial,:) = Er_or_ekf;
    
    Y_or_ekf_test = HBF_response_dec_2015( x_or_ekf , Xtest , rowY ,rowX , nf , bias); % test data
    
    Y_HBF_EKF_OR(trial,:) = Y_or_ekf_test;
    
    figure(3);
%     plot([index_of_outliers ],Youtlier,'or');hold on
    plot(Ytest,'k'),hold on;
    plot(Y_or_ekf_test,'r'),hold on
    legend('Test','HBF-EKF-OR')
    pause(.5)
trial

% Note: on display during training you can see # of interation, error and #
% of iterations of EKF OR, respectively
end

%==========================================================================
%==========================================================================

disp('RMSE_TEST MSE_TEST MAE_TEST RMSE_TRAIN MSE_TRAIN MAE_TRAIN')
'mean_HBF_OR_EKF_Parameters'
mean_HBF_OR_EKF_Parameters = mean(HBF_OR_EKF_Parameters)
'std_HBF_OR_EKF_Parameters'
std_HBF_OR_EKF_Parameters = std(HBF_OR_EKF_Parameters)

%==========================================================================
%==========================================================================
figure(3);clf
plot(Ytest,'--k'),hold on;
plot(mean(Y_HBF_EKF_OR),'r'),hold on
plot(index_of_outliers,Youtlier,'dk','markerfacecolor','k','markersize',3)
xlabel('Time index')
legend('Test','HBF-EKF-OR','outliers')
title('Average HBF-EKF-OR Output')
%==========================================================================
%==========================================================================

