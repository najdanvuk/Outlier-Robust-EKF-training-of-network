function y = gaussian(x, mu, C)

% define quadratic form
xg = x - mu; invC = (C)^-1;

% calculate "gaussian"
y = exp(-.5*xg'*invC*xg);

return
