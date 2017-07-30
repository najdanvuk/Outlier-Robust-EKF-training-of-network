
% clc,clear,close all
X1 = linspace(-1,1,300)
X2 = linspace(-1,1,300)
X = [ X1  ; X2 ];
Y = exp(X(1,:) .* sin(pi*X(2,:)));
Yclear = Y;


% =========================================================================
% =========================================================================
% 
% % simulate gross error noise model as first order Markov chain
% % define state
% delta = .05;
% DoF = 50;
% noiseVar1 = 1e-7;
% noiseVar2 = 1;
% state  = [0;...         % casual noise :)
%             1] ;         % outlier 
% % define transition probability
% P = [1-delta delta;... 
%     delta 1-delta ];
% 
% % define initial probabilty of state
% pi0 = rand(1,2);
% pi0 = pi0./sum(pi0);
% 
% [chain,state] = simulate_markov(state,P,pi0,numel(Y));
% % 
% minus_plus = [ones(1,numel(Y)/2) -ones(1,numel(Y)/2) ];
% minus_plus = minus_plus(randperm(size(minus_plus,2)));
% 
% outliers = [];
% Casual_noise = [];
% Yt = Y;
% for k = 1 : numel(Y)
%     if chain(k) == 1
%         casual_noise = iwishrnd(1,DoF); %randn(); %;
%         Yt(k) = Yt(k) + minus_plus(k) * sqrt(casual_noise) ;
%         Casual_noise = [Casual_noise ; k Yt(k)];%casual_noise ];
%     else 
%         outlier = sqrt(noiseVar2) * randn();
%         Yt(k) = Yt(k) + outlier;
%         outliers = [outliers; k Yt(k)];%outlier]];
%     end
% %     Yt = [Yt Ytr(k)]; 
% %     Ytrain = [Ytrain; train(kk+1,1)];
% end

% add outliers
pct_of_outliers = 1
number_of_outliers = round(pct_of_outliers * size(Y,2))
offset_of_outliers = .2

index_of_outliers = randi([1 size(X,2)],1,number_of_outliers)

Youtlier  = Y(:,index_of_outliers) + offset_of_outliers .* (2*rand(size(Y,1),number_of_outliers) - ones(size(Y,1),number_of_outliers))
Xoutlier = X(:,index_of_outliers)

plot([index_of_outliers ],Youtlier,'or');hold on

Y(:,index_of_outliers) = Youtlier

plot(Y)
figure(313);
plot(Y,'r');hold on;
plot(Yclear)
