% clc,clear,close all
% 
% clc,clear,close all
% nonlinear dynamical system

delta = .05
kmax = 802
% test set
% define test controls
Y=[];utest = []
for k = 1 : kmax
    if k <= 500
        test = sin(3*pi*k/250);
    else (k > 500) && (k < kmax);
        test = .25 * sin(2*pi*k/250) + .2*sin(3*k*pi/50);
    end
    utest = [utest test];
end
utest;
% and this is system output
for k = 1 : kmax
    if k == 1
        y = utest(k);
    elseif k == 2 
        y = Y(k-1) * utest(k-1) + utest(k);
    elseif k == 3
        y = ( Y(k-1) * Y(k-2) * utest(k-1)  + utest(k) ) ./ (1 + Y(k-2)^2);
    elseif k > 4
        y = ( Y(k-1) * Y(k-2) * Y(k-3) * utest(k-1) * (Y(k-3)-1) + utest(k) ) ./ (1 + Y(k-2)^2 + Y(k-3)^2) ;
    end
    Y = [Y y];
end

Xtest = []
Ytest = []

for kk = 3 : kmax
    test_vec = [Y(kk) Y(kk-1) Y(kk-2) utest(kk) utest(kk-1)]';
    Xtest = [Xtest  test_vec];
end

for kk = 1 : kmax-2
    Ytest = [Ytest  Y(kk)];
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% 
% training set
utrain = linspace(-2,2,kmax);
Ytr = []
for k = 1 : kmax
    if k == 1
        y = utrain(k);
    elseif k == 2 
        y = Ytr(k-1) * utrain(k-1) + utrain(k);
    elseif k == 3
        y = ( Ytr(k-1) * Ytr(k-2) * utrain(k-1)  + utrain(k) ) ./ (1 + Ytr(k-2)^2);
    elseif k > 4
        y = ( Ytr(k-1) * Ytr(k-2) * Y(k-3) * utrain(k-1) * (Ytr(k-3)-1) + utrain(k) ) ./ (1 + Ytr(k-2)^2 + Ytr(k-3)^2);
    end
    Ytr = [Ytr y];
end

figure(1);plot(Ytest)
% figure(2);plot(utest)

% simulate gross error noise model as first order Markov chain
% define state
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

Xtrain = []
Ytrain = []

for k = 3 : kmax
    train_vec = [Ytr(k) Ytr(k-1) Ytr(k-2) utrain(k) utrain(k-1)]';
    Xtrain = [Xtrain  train_vec];
end

outliers = [];
Casual_noise = [];
for k = 1 : kmax - 2
    if chain(k) == 1
         
        Ytr(k) = Ytr(k) + sqrt(.1) * randn(); %iwishrnd(1,10);
        Casual_noise = [Casual_noise ; k Ytr(k)];%casual_noise ];
    else 
        outlier = sqrt(1) * randn();
        Ytr(k) = Ytr(k) + outlier;
        outliers = [outliers; k Ytr(k)];%outlier]];
    end
    Ytrain = [Ytrain Ytr(k)]; 
%     Ytrain = [Ytrain; train(kk+1,1)];
end

figure(3);plot(Ytr); hold on
figure(4);plot(Ytrain,'b');hold on
plot(Casual_noise(:,1),Casual_noise(:,2),'sg');hold on
plot(outliers(:,1),outliers(:,2),'or')

figure(5); plot(Ytrain,'k'), hold on; plot(Ytest,'r')

% outliers = []
% gaussian_noise = []
% % uniform_noise = []
% for kk = 1 : size(x,2)
%     
%         gauss_noise = std_noise * randn();
%         uni_noise = std_noise * (2 * rand() - 1)
%         ynoise(kk) = y(kk) + gauss_noise %uni_noise%% ;
%         gaussian_noise = [gaussian_noise ;[kk ynoise(kk)]];
%         
%     if chain(kk) == 1
%         outlier = minus_plus(kk) * iwishrnd(1,DoF);
%         ynoise(kk) = ynoise(kk) + outlier;
%         outliers = [outliers; [kk ynoise(kk)]];  
% %     else
% %         ynoise(kk)= ynoise(kk);
%     end
%     
% end