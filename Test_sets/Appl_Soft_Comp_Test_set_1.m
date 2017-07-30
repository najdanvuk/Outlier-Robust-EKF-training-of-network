
% Radial basis function networks with hybrid learning for system
% identification with outliers
% Applied Soft Computing 11 (2011) 3083–3092
% Example # 1
% clc,clear
Y = []
U = []
kmax = 1000
ktrain = 500
for k = 1 : kmax
    u = [sin(2*pi*k/25); cos(2*pi*k/25)]
    U = [U u];
    if k == 1
        y = zeros(2,1)
    else k > 1
        y = [Y(1,k-1)/(1+(Y(2,k-1))^2); Y(1,k-1)*Y(2,k-1)/(1+(Y(2,k-1))^2)] + U(:,k-1)
    end
    Y = [Y y];
end



% training set
Xtrain = [];
Ytrain = [];
for kk = 2 : ktrain+2
    xtrain = [Y(:,kk-1) ; U(:,kk-1)];
    ytrain = Y(:,kk);
    Xtrain = [Xtrain xtrain];
    Ytrain = [Ytrain ytrain];

end
Xtrain
Ytrain
%  testing set
Xtest = [];
Ytest = [];
for kk = ktrain+1 : kmax
    xtest = [Y(:,kk-1) ; U(:,kk-1)];
    ytest = Y(:,kk);
    Xtest = [Xtest xtest];
    Ytest = [Ytest ytest];

end
Xtest
Ytest
% Xtest
% Ytest