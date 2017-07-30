clc,clear,close all

belexline_data;
[row,col] = size(belex_data);
% plot(belex_data)
step = 10;
X = [] ; Y = [];

for kk = step+1 : row-1
    index = kk-step:kk;
        xtrain = [belex_data(index)];
        X = [X  xtrain];
        Y = [Y belex_data(kk+1)];
end
clear belex_data

X = (X - min(X(:))) ./ (max(X(:)) - min(X(:)));

Y = (Y - min(Y(:))) ./ (max(Y(:)) - min(Y(:)));

% simulate gross error noise model as first order Markov chain
% define state
delta = .05;
DoF = 15;
noiseVar1 = 1;
noiseVar2 = 1e-3;
state  = [0;...         % casual noise :)
            1] ;         % outlier 
% define transition probability
P = [1-delta delta;... 
    delta 1-delta ];

% define initial probabilty of state
pi0 = rand(1,2);
pi0 = pi0./sum(pi0);

[chain,state] = simulate_markov(state,P,pi0,numel(Y));
% 
minus_plus = [ones(1,numel(Y)/2) -ones(1,numel(Y)/2) ];
minus_plus = minus_plus(randperm(size(minus_plus,2)));

outliers = [];
Casual_noise = [];
Yt = Y;
for k = 1 : numel(Y)-step
    if chain(k) == 1
        casual_noise = iwishrnd(1,DoF); %randn(); %;
        Yt(k) = Yt(k) + minus_plus(k) * sqrt(casual_noise) ;
        Casual_noise = [Casual_noise ; k Yt(k)];%casual_noise ];
    else 
        outlier = sqrt(noiseVar2) * randn();
        Yt(k) = Yt(k) + outlier;
        outliers = [outliers; k Yt(k)];%outlier]];
    end
%     Yt = [Yt Ytr(k)]; 
%     Ytrain = [Ytrain; train(kk+1,1)];
end

figure(313);
plot(Yt,'r');hold on;
plot(Y)