function [ x , Er , Statistics ] = RBF_EKF_Decembar_2013( Xtrain , Ytrain , Xtest , Ytest , parameters)


% Radial basis function neural network training.
% Functions are Gaussains.
% The learning is based on extende Kalman Filer.

% The file operates as a sequential learning algorithm, i.e. it sequentially takes
% xn,yn from training set, process it and discards it afterwards.

% INPUTS
% x = training inputs, an ni x M matrix, where
%     ni is the dimension of each input, and
%     M is the total number of training vectors.
% y = training outputs, an no x M matrix, where
%     no is the dimension of each output, and
%     M is the total number of training vectors.
% nf = # of radial basis function centers.
% epsilon = delta-error threshold at which to stop training.
% P0 = initial setting of estimate covariance matrix.
% Q0 = initial setting of state covariance matrix.
% R0 = initial setting of measurement covariance matrix.
%
% OUTPUTS
% Centers = centres of RBFs ni x nf matrix.
% Weights = weight matrix between middle layer and output layer, an no x (nf+1) matrix.
% Spreads = widths of basis functions.

%==========================================================================
%==========================================================================

nf = parameters.number_of_functions;
xS0 = parameters.xS0 ;
initVar = parameters.initVar; 
p0 = parameters.p0;            % inital state uncertainty
q0 = parameters.q0 ;            % process covariance
r0 = parameters.r0 ;           % measurement covariance
ne = parameters.ne;              % % pct of data to be used in training

%==========================================================================
%==========================================================================
%                              T R A I N I N G
% =========================================================================
%==========================================================================
[rowX,colX] = size(Xtrain);                   % dimension of input vector
no = size(Ytrain,1);                         % dimension of output vector
% M = round(ne * colX)                    % number of training examples
%==========================================================================
%==========================================================================
% initialize centers of RBF Gaussians (choose randomly from training set)
for i = 0 : nf-1
    xC(:, i+1) = Xtrain(:, round(colX*i/nf) + 1);
end
xS = ones(1,nf) * xS0;           % initialize spreads of RBF Gaussians
% =========================================================================
% Define state vector and other system parameters (P,Q,R)
W = sqrt(initVar).*randn(nf,no);         % inital value of weights (number of RBFs) x (dimension of output vector)
Wbias = sqrt(initVar) .* randn(no,1) ;   % bias
n = nf * (rowX + 1 ) + no * ( nf + 1);  % dimension of state vector
P = p0 * eye(n);                        % inital system uncertainty
Q = q0 * eye(n);                        % process/system uncertainty
R = r0 * eye(no);                       % measurement uncertainty
%==========================================================================
% =========================================================================
% Define state vector
x = [Wbias ; W(:) ; xC(:) ; xS'];
% =========================================================================
% =========================================================================
% Record some data
Er = [];

for sample = 1 : colX
    
    % Compute the partial derivative of the error with respect to
    % the components of the prototypes in the v matrix.
    [xB, xW, xC, xS] = extract_RBF_network_parameters( x , no , rowX , nf ) ;
    
    H = RBF_JACOBIAN_Dec_2013(Xtrain(:,sample),xS,xC,xW,no);
    
    inovvector = Ytrain(:,sample) - RBF_response_dec_15(x,Xtrain(:,sample),no,rowX,nf);
    
    % Compute the Kalman gain.
    invS = inv( R + H' * P * H);
    K = P * H * invS;
    % Update the state vector estimate.
    x  = x + K * inovvector ;
    % Update the covariance matrix.
    P  = P  - K * H' * P + Q;
    
    % Extract weights, centers and widths from state vector x.
    [xB, xW, xC, xS] = extract_RBF_network_parameters(x,no,rowX,nf);
    % compute error
    %         E = .5 * (residual)'*(residual)
    E = .5 * (Ytrain(:,sample) - RBF_response_dec_15(x,Xtrain(:,sample),no,rowX,nf))*...
        (Ytrain(:,sample) - RBF_response_dec_15(x,Xtrain(:,sample),no,rowX,nf))';
    
    Er = [Er E];
    disp([,num2str(sample),' ; ',num2str(E)])
    
end

Yekf_test = RBF_response_dec_15(x,Xtest,no,rowX,nf); % test data
Yekf_train = RBF_response_dec_15(x,Xtrain,no,rowX,nf);   % train data

% gather statistics
MSE_TEST = mse(Yekf_test - Ytest);
RMSE_TEST = sqrt(MSE_TEST);
MAE_TEST = mae(Yekf_test - Ytest);

MSE_TRAIN = mse(Yekf_train-Ytrain);
RMSE_TRAIN = sqrt(MSE_TRAIN);
MAE_TRAIN = mae(Yekf_train-Ytrain);

Statistics = [RMSE_TEST MSE_TEST MAE_TEST RMSE_TRAIN MSE_TRAIN MAE_TRAIN];

