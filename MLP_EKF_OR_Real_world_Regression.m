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
% Training of MLP: EKF vs EKF-OR for real finance data.
% Important to note: I am not trying to model the price of asset itself but
% rather to reconstruct unknown signal from noisy data. Therefore, the point
% of this code/model is just to see how MLP trained with EKF OR deals with
% stochastic function contaminated with outliers. Had I desired to model
% returns of these assets I would have taken entirely different approach.
% %========================================================================
% %========================================================================
clc,clear,close all
% rng('default')
% %========================================================================
% %========================================================================
% =========================================================================
% =========================================================================
%                           T E S T   S E T S
% =========================================================================
% =========================================================================
% Microsoft_data
% data = Microsoft;
% clear Microsoft
% =========================================================================
% =========================================================================
% Google_data
% data = Google;
% clear Google
% =========================================================================
% =========================================================================
% RedHat_data
% data = RedHat
% clear RedHat
% =========================================================================
% =========================================================================
% Intel_data
% data = Intel;
% clear Intel
% =========================================================================
% =========================================================================
% IBM_data
% data = IBM;
% clear IBM
% % =========================================================================
% =========================================================================
% SNP500_data
% data = SNP500;
% clear SNP500
% =========================================================================
% =========================================================================
% =========================================================================
% =========================================================================
% Oracle_data
% data = Oracle;
% clear Oracle
% =========================================================================
% =========================================================================
% Yahoo_data
% data = Yahoo;
% clear Yahoo
% =========================================================================
% =========================================================================
Apple_data
data = Apple;
clear Apple
% =========================================================================
% =========================================================================
% belexline_data
% data = belex_data;
% clear belex_data
% =========================================================================
% =========================================================================
[row,col] = size(data);
% % scale data
[data , settings] = mapminmax(data',-1,1);   % to [-1,1]
% data = (data - min(data))./(max(data)-min(data));
data = data';
% =========================================================================
% =========================================================================
%              I N I T I A L I Z E   P A R A M E T E R S
% =========================================================================
% =========================================================================
ne = 1 %% pct of data to be used in training
number_of_trials = 10  % number of independent training runs/trials
%==========================================================================
%==========================================================================
% Define basic network parameters
hidden_layers = [10 5]    % number of layers and neurons: e.g. [10] => one laye with ten neurons
% [10 5] => two layers with 10 and five neurons respectivly
weightVar = .01         % inital variance for weights initialization
biasVar = .001          % inital variance for bias initialization
p0 = 1e0                % inital state uncertainty
q0 = 1e0                % process covariance
r0 = 1e-1               % measurement covariance
%==========================================================================
%==========================================================================
% % simulate gross error noise model as first order Markov chain
% % define state
delta = .05;
DoF_h = 10;
noiseVar1 = 1e-5;
noiseVar2 = 1e-2;
state  = [0;...         % casual noise :)
    1] ;         % outlier

% define transition probability
P = [1-delta delta;...
    delta 1-delta ];

% define initial probabilty of state
pi0 = rand(1,2);
pi0 = pi0./sum(pi0);

[chain,state] = simulate_markov(state,P,pi0,numel(data));
%
minus_plus = [ones(1,round(numel(data)/2)) -ones(1,round(numel(data)/2)) ];
minus_plus = minus_plus(randperm(size(minus_plus,2)));

data_outliers = data;
Outliers = [];
Casual_noise = [];
for k = 1 : numel(data)
    if chain(k) == 0;
        casual_noise = iwishrnd(1,DoF_h);%randn * sqrt(noiseVar1);%;
        data_outliers(k) = data_outliers(k) + (casual_noise) ;
        Casual_noise = [Casual_noise ; k data_outliers(k)];
    else chain(k) == 1;
        outlier = sqrt(noiseVar2) * randn();
        data_outliers(k) = data_outliers(k) + outlier;
        Outliers = [Outliers; k data_outliers(k)];
    end
end
%==========================================================================
%==========================================================================

figure(12);
close all
plot(data,'b');hold on;
plot(data_outliers,'r')

time_series_step = 3;       % I just assume that three previous states are
% enough but for thorough estimation PACF analysis is needed

% form training set
Xtrain = [] ; Ytrain = [];
for kk = time_series_step+1 : row-1
    index = kk-time_series_step:kk-1;
    xtrain = [data_outliers(index)];
    Xtrain = [Xtrain xtrain];
    Ytrain = [Ytrain data_outliers(kk)];
end
Ytrain_real = Ytrain; % Ytrain for data ploting
% form testing set using free of outliers
Xtest = [] ; Ytest = [];
for kk = time_series_step+1 : row-1
    index = kk-time_series_step:kk-1;
    xtest = [data(index)];
    Xtest = [Xtest xtest];
    Ytest = [Ytest data(kk)];
end

% =========================================================================
% =========================================================================
figure(313); close all
plot(Ytrain,'r','linewidth',2);hold on;
% plot(index_of_outliers,Youtlier,'sk','markersize',2,'markerfacecolor','k'),hold on
plot(Ytest,'b','linewidth',2);hold on
% plot(Outliers(:,1),Outliers(:,2),'sr','markersize',10,'markerfacecolor','r','markersize',3);hold on
% plot(Casual_noise(:,1),Casual_noise(:,2),'db','markersize',3,'markerfacecolor','b')
legend('Train','Test','Outliers','Noise')
%==========================================================================
%==========================================================================
% figure(317);
% plot(1:numel(Ytrain),Ytrain,'r','linewidth',2);hold on;
% % plot(index_of_outliers,Youtlier,'sk','markersize',2,'markerfacecolor','k'),hold on
% plot(numel(Ytrain)+1:numel(Ytrain)+numel(Ytest),Ytest,'b','linewidth',2);hold on
% % plot(Outliers(:,1),Outliers(:,2),'sr','markersize',10,'markerfacecolor','r','markersize',3);hold on
% % plot(Casual_noise(:,1),Casual_noise(:,2),'db','markersize',3,'markerfacecolor','b')
% legend('Train','Test','Outliers','Noise')

%==========================================================================
%==========================================================================
[ rowY , colY ] = size(Ytrain);
[ rowX , colX ] = size(Xtrain);
ne_times_colY = round(ne * colY);

MLP_EKF_Parameters = zeros(number_of_trials,6);
MLP_EKF_Error = zeros(number_of_trials,ne_times_colY);
MLP_OR_EKF_Parameters = zeros(number_of_trials,6);
MLP_OR_EKF_Error = zeros(number_of_trials,ne_times_colY);

PCT_EKF = [];
PCT_OR_EKF = [];

Y_EKF = zeros(number_of_trials,colY)
Y_OR_EKF = zeros(number_of_trials,colY)

%==========================================================================
%==========================================================================
% EKF

mlp_ekf_parameters.p0  = p0;
mlp_ekf_parameters.q0 = q0;
mlp_ekf_parameters.r0	= r0;
mlp_ekf_parameters.hidden_layers = hidden_layers;
mlp_ekf_parameters.weightVar = weightVar;
mlp_ekf_parameters.biasVar = biasVar ;
mlp_ekf_parameters.ne = ne;

% EKF OR
mlp_or_ekf_parameters.p0  = p0;
mlp_or_ekf_parameters.q0 = q0;
mlp_or_ekf_parameters.r0	= r0;
mlp_or_ekf_parameters.hidden_layers = hidden_layers;
mlp_or_ekf_parameters.weightVar = weightVar;
mlp_or_ekf_parameters.biasVar = biasVar ;
mlp_or_ekf_parameters.ne = ne;
mlp_or_ekf_parameters.DoF_h = DoF_h ;

%==========================================================================
%==========================================================================
%                              T R A I N I N G
% =========================================================================
%==========================================================================

for step = 1 : number_of_trials
    clc
    step
    %==========================================================================
    %==========================================================================
    % Prepare data for sequential training
    % Randomize data for training
    N = randomize_data(colY, round(ne_times_colY));
    
    % Define training and testing set
    Xtrain = Xtrain(:,N);Ytrain = Ytrain(:,N);
    
    %==========================================================================
    %==========================================================================
    % =========================================================================
    %==========================================================================
    
    disp(num2str(step))
    %     EKF
    [ mlp_ekf , Er_ekf , Statistics_ekf , Ymlp_test_ekf ] = MLP_EKF_training_Decembar_2013( Xtrain , Ytrain , Xtest , Ytest , mlp_ekf_parameters );
    MLP_EKF_Parameters(step,:) = Statistics_ekf;
    MLP_EKF_Error(step,:) = Er_ekf;
    
    %     EKF OR
    [ mlp_or_ekf , Er_or_ekf , Statistics_or_ekf , Ymlp_test_or_ekf , invS , Ress] = MLP_robust_EKF_training_Decembar_2013( Xtrain , Ytrain , Xtest , Ytest , mlp_or_ekf_parameters );
    
    MLP_OR_EKF_Parameters(step,:) = Statistics_or_ekf;
    MLP_OR_EKF_Error(step,:) = Er_or_ekf;
    
    Y_EKF(step,:) = Ymlp_test_ekf;
    Y_OR_EKF(step,:) = Ymlp_test_or_ekf;
    
    figure(3);clf
    plot(Ytest,'k'),hold on;
    plot(Ymlp_test_ekf,'b'),hold on
    plot(Ymlp_test_or_ekf,'r'),hold on
    legend('Test','MLP-EKF','MLP-EKF-OR')
    pause(1)
end
%==========================================================================
%==========================================================================

disp('RMSE_TEST MSE_TEST MAE_TEST RMSE_TRAIN MSE_TRAIN MAE_TRAIN')
'mean_MLP_EKF_Parameters '
mean_MLP_EKF_Parameters = mean(MLP_EKF_Parameters)
'std_MLP_EKF_Parameters '
std_MLP_EKF_Parameters = std(MLP_EKF_Parameters)
'mean_MLP_OR_EKF_Parameters'
mean_MLP_OR_EKF_Parameters = mean(MLP_OR_EKF_Parameters)
'std_MLP_OR_EKF_Parameters'
std_MLP_OR_EKF_Parameters = std(MLP_OR_EKF_Parameters)

mean_MLP_EKF_Error = mean(MLP_EKF_Error) ;
mean_MLP_OR_EKF_Error = mean(MLP_OR_EKF_Error) ;

Improvement_rate = (mean_MLP_EKF_Parameters(1) - mean_MLP_OR_EKF_Parameters(1)) / mean_MLP_EKF_Parameters(1) *100

%==========================================================================
%==========================================================================

figure(19),clf
subplot(2,1,1);
loglog(1:numel(mean_MLP_EKF_Error),mean_MLP_EKF_Error);hold on
loglog(1:numel(mean_MLP_OR_EKF_Error),mean_MLP_OR_EKF_Error);
subplot(2,1,2);
plot(1:numel(mean_MLP_EKF_Error),mean_MLP_EKF_Error);hold on
plot(1:numel(mean_MLP_OR_EKF_Error),mean_MLP_OR_EKF_Error);

figure(3);clf
% subplot(2,1,2)
plot(Ytest,'k'),hold on;
plot(mean(Y_EKF),'b'),hold on
plot(mean(Y_OR_EKF),'r')
legend('Test','MLP-EKF','MLP-EKF-OR')

figure(322);
boxplot([MLP_EKF_Parameters(:,1) MLP_OR_EKF_Parameters(:,1)],'plotstyle','traditional',...  %'notch','on',...
    'labels',{'EKF','EKF-OR'})

% =========================================================================
% =========================================================================
% plots for paper
% firstly, transform from [-1,1] to real data space
mean_Y_OR_EKF = mean(Y_OR_EKF); mean_Y_EKF = mean(Y_EKF);
% Ytest = mapminmax('reverse',Ytest,settings);
% Ytrain_real = mapminmax('reverse',Ytrain_real,settings);
% mean_Y_OR_EKF = mapminmax('reverse',mean_Y_OR_EKF,settings);
% mean_Y_EKF = mapminmax('reverse',mean_Y_EKF,settings);


figure(731); clf
subplot(211);
plot(mapminmax('reverse',Ytrain_real,settings),'r','linewidth',2);hold on;
% plot(index_of_outliers,Youtlier,'sk','markersize',2,'markerfacecolor','k'),hold on
plot( mapminmax('reverse',Ytest,settings),'b','linewidth',2);hold on
% plot(Outliers(:,1),Outliers(:,2),'sr','markersize',10,'markerfacecolor','r','markersize',3);hold on
% plot(Casual_noise(:,1),Casual_noise(:,2),'db','markersize',3,'markerfacecolor','b')
legend('Train','Test','Outliers','Noise')
subplot(212)
plot( mapminmax('reverse',Ytest,settings),'.-k'),hold on;
% plot(mean(Y_EKF),'b'),hold on
plot(mapminmax.reverse(mean(Y_OR_EKF),settings),'r','linewidth',2);hold on
legend('Test','MLP-EKF-OR')
% subplot(313)
% boxplot([MLP_EKF_Parameters(:,1) MLP_OR_EKF_Parameters(:,1)],'plotstyle','traditional',...  %'notch','on',...
%     'labels',{'EKF','EKF-OR'})


figure(11313); clf
set(gca,'FontName','Arial','FontSize',14);
subplot(221);
plot(Ytrain_real,'--k','linewidth',2);hold on;
% plot(index_of_outliers,Youtlier,'sk','markersize',2,'markerfacecolor','k'),hold on
plot(Ytest,'b','linewidth',2);hold on
% plot(Outliers(:,1),Outliers(:,2),'sr','markersize',10,'markerfacecolor','r','markersize',3);hold on
% plot(Casual_noise(:,1),Casual_noise(:,2),'db','markersize',3,'markerfacecolor','b')
xlabel('Time index');ylabel('')
legend('Train','Test','Outliers','Noise')
subplot(223)
plot(Ytest,'.-k'),hold on;
% plot(mean(Y_EKF),'b'),hold on
plot(mean_Y_OR_EKF,'r','linewidth',2)
plot(mean_Y_EKF,'b','linewidth',2)
xlabel('Time index');ylabel('')
legend('Test','MLP-EKF-OR','MLP-EKF')
subplot(2,2,[2 4])
boxplot([MLP_EKF_Parameters(:,1) MLP_OR_EKF_Parameters(:,1)],'plotstyle','traditional',...  %'notch','on',...
    'labels',{'EKF','EKF-OR'})
ylabel('RMSE')

% =========================================================================
% =========================================================================
% Finally, let us do some statistical hypothesis testing. For this purpose
% we use Wilcoxon signed rank test and we test difference between two
% samples: RMSE of EKF MLP and RMSE of MLP EKF OR.
[p,h] = signrank(MLP_EKF_Parameters(:,1),MLP_OR_EKF_Parameters(:,1))
% and as this test indicates, if h=0 we cannot reject null hypothesis of 
% equality between two samples, therefore, all this is for nothing :) ...

%==========================================================================
%==========================================================================



