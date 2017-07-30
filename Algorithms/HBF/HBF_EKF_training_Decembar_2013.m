function [ x , Er , Statistics ] = HBF_EKF_training_Decembar_2013(Xtrain , Ytrain , Xtest , Ytest , parameters)

% Hyper basis function neural network training.
% Functions are diagonal Gaussains.
% The file operates as a sequential learning algorithm, i.e. it sequentially takes
% xn,yn from training set, process it and discards it afterwards.

% To be more specific, unlike HBF-GP case, in this setup we use HBF neural
% network as genuine RBF network, withou ability to add or delete neurons.

% INPUTS
% x = training inputs, an ni x M matrix, where
%     ni is the dimension of each input, and
%     M is the total number of training vectors.
% y = training outputs, an no x M matrix, where
%     no is the dimension of each output, and
%     M is the total number of training vectors.
% nf = # of hyper basis function centers.
% epsilon = delta-error threshold at which to stop training.
% P0 = initial setting of estimate covariance matrix.
% Q0 = initial setting of state covariance matrix.
% R0 = initial setting of measurement covariance matrix.
%
% OUTPUTS
% Centers = centres of HBFs ni x nf matrix.
% Weights = weight matrix between middle layer and output layer, an nf x no matrix.
% Spreads = widths of basis functions given as a vector; each element for
% one dimension.

%==========================================================================
%==========================================================================
[rowX,colX] = size(Xtrain); [ rowY ,colY] = size(Ytrain);

% Define basic parameters
nf = parameters.number_of_functions;
xS0 = parameters.xS0 ;
initVar = parameters.initVar; 
initBias = parameters.initBias;
p0 = parameters.p0;            % inital state uncertainty
q0 = parameters.q0 ;            % process covariance
r0 = parameters.r0 ;           % measurement covariance
ne = parameters.ne;              % % pct of data to be used in training
bias = parameters.bias;
Emin = parameters.Emin;
%==========================================================================
%==========================================================================

% number of training examples
M = round(ne * colX);
% Randomize data for training
% N = randomize_data(colX, M);

% initialize centers of RBF Gaussians (choose randomly from training set)
for i = 0 : nf-1
    xC(:, i+1) = Xtrain(:, round(M*i/nf) + 1);
end
xS = ones(rowX,nf) * xS0 ;          % initialize spreads of RBF Gaussians
%==========================================================================
%==========================================================================
%                              T R A I N I N G
% =========================================================================
%==========================================================================
switch bias
    case 'no_bias'
        
        % Define state vector and other system parameters (P,Q,R)
        W = sqrt(initVar).*randn(nf,rowY);         % inital value of weights (number of RBFs) x (dimension of output vector)
        dim_x = nf * ( 2 * rowX + rowY );                 % dimension of state vector

        P = p0 * eye(dim_x);                                 % state uncertainty
        Q = q0 * eye(dim_x);                            % system uncertainty
        R = r0 * eye(rowY);                           % measurement uncertainty
        %==========================================================================
        % =========================================================================
        % Define state vector
        x = [W(:) ; xC(:); xS(:)];
        % =========================================================================
        % =========================================================================
    case 'with_bias'
        
        % Define state vector and other system parameters (P,Q,R)
        W = sqrt(initVar).*randn(nf,rowY);         % inital value of weights (number of RBFs) x (dimension of output vector)
        Wbias = sqrt(initBias) .* randn(rowY,1);    % bias
        dim_x = nf * ( 2 * rowX + rowY ) + rowY;                 % dimension of state vector
        P = p0 * eye(dim_x);                                 % state uncertainty
        Q = q0 * eye(dim_x);                            % system uncertainty
        R = r0 * eye(rowY);                           % measurement uncertainty
        %==========================================================================
        % =========================================================================
        % Define state vector
        x = [Wbias ; W(:) ; xC(:); xS(:)];
        % =========================================================================
        % =========================================================================
end

% Record some data
Er = [];
% Rrecord = [];

for sample = 1 : M
    
    [ xB , xW , xC , xS ] = extract_HBF_network_parameters( x , rowY,rowX,nf,bias);
    
    H = HBF_jacobian_Dec_2013(Xtrain(:,sample),xW,xC,xS,rowY,bias);
    
    inovvector = Ytrain(:,sample) -  HBF_response_dec_2015( x , Xtrain(:,sample),rowY,rowX,nf, bias);
    %
    % Compute the Kalman gain.
    invS = inv( R + H' * P * H);
    K = P * H * invS;
    % Update the state vector estimate.
    x  = x + K * inovvector ;
    % Update the covariance matrix.
    P  = P  - K * H' * P + Q;
    %     % compute error
    E = .5 * (Ytrain(:,sample) - HBF_response_dec_2015(x,Xtrain(:,sample),rowY,rowX,nf,bias))*...
        (Ytrain(:,sample) - HBF_response_dec_2015(x,Xtrain(:,sample),rowY,rowX,nf,bias))';
    %
    Er = [Er E];
    
    disp([,num2str(sample),' ; ',num2str(E)])
    if E < Emin
        break
    end
end
% %
%
Yekf_test = HBF_response_dec_2015(x,Xtest,rowY,rowX,nf,bias); % test data
Yekf_train = HBF_response_dec_2015(x,Xtrain,rowY,rowX,nf,bias);   % train data
%
% % gather statistics
MSE_TEST = mse(Yekf_test - Ytest);
RMSE_TEST = sqrt(MSE_TEST);
MAE_TEST = mae(Yekf_test - Ytest);

MSE_TRAIN = mse(Yekf_train - Ytrain);
RMSE_TRAIN = sqrt(MSE_TRAIN);
MAE_TRAIN = mae(Yekf_train - Ytrain);

disp('RMSE_TEST MSE_TEST MAE_TEST RMSE_TRAIN MSE_TRAIN MAE_TRAIN')
Statistics = [RMSE_TEST MSE_TEST MAE_TEST RMSE_TRAIN MSE_TRAIN MAE_TRAIN];

