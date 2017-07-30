function y = gaussian_RBF(x, mu,sigma)

% define quadratic form
xg = x - mu; 
% calculate "gaussian"
y = exp(-.5*xg'*xg/sigma^2);

return
