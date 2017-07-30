function g = act_fun_derivative(x,type)
% calculates derivative of activation function
switch type
    case 'log_sig'
        g = exp(-x)/((1 + exp(-x)).^2);
    case 'tan_sig'
%         g = 4 * exp(-2*x)./(1+exp(-2 * x)).^2;
        g = 1 - tanh(x).^2;
    case 'tan_sig_2'
        g = 2 * exp(-x)./(1 + exp(-x)).^2;
end
return