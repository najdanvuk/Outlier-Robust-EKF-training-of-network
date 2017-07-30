function [ mlp , Er , Statistics , Ymlp_test ] = MLP_EKF_training_Decembar_2013( Xtrain , Ytrain , Xtest , Ytest , parameters )

% MLP neural network training.
% Functions are hyperbolic tangent.
%
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
% weights
% biases
% Error
% Statistics

%==========================================================================
%==========================================================================
P0 = parameters.p0;  % inital state uncertainty
Q0 = parameters.q0;  % process covariance
R0 = parameters.r0;	% measurement covariance
hidden_layers = parameters.hidden_layers;
weightVar = parameters.weightVar;
biasVar = parameters.biasVar ;
ne = parameters.ne ;
%==========================================================================
%==========================================================================
[rowX,colX] = size(Xtrain);                   % dimension of input vector
[rowY,colY] = size(Ytrain);                   % dimension of output vector
%==========================================================================
%==========================================================================
% =========================================================================
%==========================================================================
lengths = [rowX rowY];
% =========================================================================
%generate MLP network
mlp = gen_MLP_dec_2013( lengths , hidden_layers , weightVar , biasVar );
% =========================================================================
% Define state vector for EKF training
numlay = size(mlp,2);
% Rearrange weights
xW = [];
for k = 1 : numlay
    xW = [xW; mlp(k).weights(:)];
end
for k = 1 : numlay-1
    xW = [xW; mlp(k).biases];
end
% =========================================================================
% =========================================================================
% Define state vector and other system parameters
% dimension of state vector
if numel(hidden_layers) == 1;
    n = (rowX + rowY ) * hidden_layers + hidden_layers;
else numel(hidden_layers) == 2;
    n = rowX * hidden_layers(1) + hidden_layers(1) * hidden_layers(2) + hidden_layers(2) * rowY + sum(hidden_layers);
end
P = P0 * eye(n);                            % inital state uncertainty
Q = Q0 * eye(n);                            % system uncertainty
R = R0 * eye(rowY);                         % measurement uncertainty
%==========================================================================
%==========================================================================
% Start training
Er = [];


for step = 1 : colY
    
    % introduce new training example {Xtrain(:,step),Ytrain(:,step)}
    H = MLP_Jacobian_december_2013(lengths, hidden_layers,mlp,Xtrain(:,step));
    
    % Compute the Kalman gain.
    %     HtP = H'*P;
    %     HtPH = HtP*H;
    %     S = R + HtPH;
    invS = inv( R + H' * P * H );
    PH = P * H;
    K = PH * invS;
    % Update the state estimate.
    xW = xW + K * (Ytrain(:,step) - MLP_response(mlp, Xtrain(:,step))) ;

    % Update the covariance matrix.
    P = P - K * H' * P + Q;
    
    % Extract weights from state vector x.
    mlp = extract_MLP_parameters_from_EKF(xW,hidden_layers,lengths);
    
    residual = Ytrain(:,step) - MLP_response(mlp, Xtrain(:,step));

    E = .5 * (residual)'*(residual);
    
    Er = [Er E];
    
    disp([,num2str(step),' ; ',num2str(E)])
    
end
% end

% Calculate statistics
% testing
mlptest = mlp;
Ymlp_test = zeros(size(Ytest,1),size(Ytest,2));
for kk = 1 : size(Xtest,2)
    Ymlp_test(:,kk) = MLP_response(mlptest, Xtest(:,kk));
end

% training
Ymlp_train = zeros(size(Ytrain,1),size(Ytrain,2));
mlptrain = mlp;
for kk = 1 : size(Xtrain,2)
    Ymlp_train(:,kk) = MLP_response(mlptrain, Xtrain(:,kk));
end
% gather statistics
% Test set
MSE_TEST = mse(Ymlp_test - Ytest);
RMSE_TEST = sqrt(MSE_TEST);
MAE_TEST = mae(Ymlp_test - Ytest);
% Train set
MSE_TRAIN = mse(Ymlp_train - Ytrain);
RMSE_TRAIN = sqrt(MSE_TRAIN);
MAE_TRAIN = mae(Ymlp_train - Ytrain);

% disp('RMSE_TEST MSE_TEST MAE_TEST ')
Statistics = [RMSE_TEST MSE_TEST MAE_TEST RMSE_TRAIN MSE_TRAIN MAE_TRAIN ];
%

%
