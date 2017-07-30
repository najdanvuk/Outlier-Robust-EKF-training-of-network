% clc,clear,close all
% Radial basis function networks with hybrid learning for system
% identification with outliers
% Applied Soft Computing 11 (2011) 3083–3092
% Example # 2

Y = [];
u = [];
kmax = 100
ktrain = kmax / 2


for k = 1 : kmax
    u = [u sin(2*pi*k/25)];
    if k == 1;
        y = 0;
    elseif k == 2;
        y = (Y(k-1) *(Y(k-1) + 2.5)) / (1 + (Y(k-1))^2 ) + u(k-1);
    else k > 2;
        y = (Y(k-1) * Y(k-2) *(Y(k-1) + 2.5)) / (1 + (Y(k-1))^2 + (Y(k-2))^2) + u(k-1);
    end
    Y = [Y y];
end
figure(44)
plot(Y,'--r','linewidth',5)


% training set
Xtrain = [];
Ytrain_real = Y(1:ktrain)
for kk = 1 : ktrain
    if kk == 1;
        xtrain = zeros(3,1);
    elseif kk == 2;
        xtrain = [Y(kk-1) ; 0 ; u(k-1)];
    else kk >= 3;
        xtrain = [Y(kk-1) ; Y(kk-2) ; u(k-1)];
    end
    Xtrain = [Xtrain xtrain];
end

%  testing set
Xtest = [];
Ytest = Y(ktrain-2:kmax)
for kk = ktrain-2 : kmax
    
    if kk == 1;
        xtest = zeros(3,1);
    elseif kk == 2;
        xtest = [Y(kk-1) ; 0 ; u(k-1)];
    else kk >= 3;
        xtest = [Y(kk-1) ; Y(kk-2) ; u(k-1)];
    end
    Xtest = [Xtest xtest];
end
Xtest
Ytest
%

plot(ktrain-2 : kmax,Ytest,'r',1:numel(Ytrain_real),Ytrain_real,'b')
legend('test','train')

if switch_noise == 1
    Ytrain = Ytrain_real
    % simulate gross error noise model as first order Markov chain
    % define state
    delta = .05;
    DoF = 15;
    noiseVar2 = 1;
    state  = [0;...         % casual noise :)
        1] ;         % outlier
    % define transition probability
    P = [1-delta delta;...
        delta 1-delta ];
    
    % define initial probabilty of state
    pi0 = rand(1,2);
    pi0 = pi0./sum(pi0);
    
    [chain,state] = simulate_markov(state,P,pi0,numel(Ytrain));
    %
    minus_plus = [ones(1,numel(Ytrain)/2) -ones(1,numel(Ytrain)/2) ];
    minus_plus = minus_plus(randperm(size(minus_plus,2)));
    
    outliers = [];
    Casual_noise = [];
    for k = 1 : numel(Ytrain)
        if chain(k) == 1
            casual_noise = iwishrnd(1,DoF); %randn(); %;
            Ytrain(k) = Ytrain(k) + minus_plus(k) * sqrt(casual_noise) ;
            Casual_noise = [Casual_noise ; k Ytrain(k)];%casual_noise ];
        else
            outlier = sqrt(noiseVar2) * randn();
            Ytrain(k) = Ytrain(k) + outlier;
            outliers = [outliers; k Ytrain(k)];%outlier]];
        end
        %     Yt = [Yt Ytr(k)];
        %     Ytrain = [Ytrain; train(kk+1,1)];
    end
    
else switch_noise == 0
    Ytrain = Ytrain_real
%     add outliers
    pct_of_outliers = .03
    number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
    offset_of_outliers = 3
    
    index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
    
    Youtlier  = Ytrain(:,index_of_outliers) + sqrt(offset_of_outliers) .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers)); % randn(size(Ytrain,1),number_of_outliers) ; %
    % Xoutlier = X(:,index_of_outliers);
    
    plot([index_of_outliers ],Youtlier,'or');hold on
    
    Ytrain(:,index_of_outliers) = Youtlier;
    
end


figure(313);
plot(Ytrain,'r');hold on;
plot(Ytrain_real,'b');hold on
plot(Ytest,'k')
