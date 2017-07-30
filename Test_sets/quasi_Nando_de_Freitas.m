clc,clear,close all

N = 300;               % Number of time steps.
t = 1:1:N;             % Time.  
x0 = 0.1;              % Initial input.
x = zeros(N,1);        % Input observation.
y = zeros(N,1);        % Output observation.
x(1,1) = x0;           % Initia input. 
actualR = 3;           % Measurement noise variance.
actualQ = .01;         % Process noise variance.        

% GENERATE PROCESS AND MEASUREMENT NOISE:
% ======================================

v = sqrt(actualR)*sin(0.05*t)'.*randn(N,1);
w = sqrt(actualQ)*randn(N,1); 
% ======================================

% GENERATE INPUT-OUTPUT DATA:
% ==========================
y(1,1) = (x(1,1)^(2))./20 + v(1,1); 
yreal(1,1) = (x(1,1)^(2))./20 ;
xreal(1,1) = x(1,1);
for t=2:N,
  xreal(t,1) = 0.5*xreal(t-1,1) + 25*xreal(t-1,1)/(1+xreal(t-1,1)^(2)) + 8*cos(1.2*(t-1));
  yreal(t,1) = (xreal(t,1).^(2))./20 ;
  
    x(t,1) = 0.5*x(t-1,1) + 25*x(t-1,1)/(1+x(t-1,1)^(2)) + 8*cos(1.2*(t-1)) + w(t,1);
  y(t,1) = (x(t,1).^(2))./20 + v(t,1) ;
end;

% NOW, ADD OUTLIERS TO THE MEASUREMENT NOISE
% ========================================================================
% add some noise and introduce outliers in training set
delta = .05
DoF = 25;
% simulate bistable burst noise (first order Markov chain)
% define state
state  = [0;...         % burst noise
            1]          % standard noise
% define transition probability
P = [1-delta delta;... 
    1-delta delta];
% define initial probabilty of state
pi0 = [.5 .5];
pi0 = pi0./sum(pi0);
[chain,state] = simulate_markov(state,P,pi0,N);
minus_plus = [ones(1,N/2) -ones(1,N/2) ];
minus_plus = minus_plus(randperm(size(minus_plus,2)));

outliers = []
gaussian_noise = []
% uniform_noise = []
xnew(1,1) = x0;
ynew(1,1) = (x(1,1)^(2))./20 + v(1,1);

for t=2:N
        xnew(t,1) = 0.5*xnew(t-1,1) + 25*xnew(t-1,1)/(1+xnew(t-1,1)^(2)) + 8*cos(1.2*(t-1)) + w(t,1);
        ynew(t,1) = (xnew(t,1).^(2))./20;
        if chain(t) == 1;
            ynew(t,1) = ynew(t,1) + iwishrnd(1,DoF);%minus_plus(t) * 
        else chain(t) == 0;
            ynew(t,1) = ynew(t,1) + v(t,1) ;
        end
end

figure(33)
plot(ynew,'r')
hold on
plot(yreal,'b')
hold on
plot(y,'k')
legend('invW','Gauss noise','without')
% figure(11);
% subplot(221)
% plot(xnew)
% ylabel('Input','fontsize',15);
% xlabel('Time','fontsize',15);
% subplot(222)
% plot(ynew)
% ylabel('Output','fontsize',15);
% xlabel('Time','fontsize',15);
% subplot(223)
% plot(x)
% ylabel('Input','fontsize',15);
% xlabel('Time','fontsize',15);
% subplot(224)
% plot(y)
% ylabel('Output','fontsize',15);
% xlabel('Time','fontsize',15);

