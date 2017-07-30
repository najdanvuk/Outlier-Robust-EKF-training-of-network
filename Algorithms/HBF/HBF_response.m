function y = HBF_response(xW,X,no,ni,nf,bias_neuron)

[ bias , xW, xC, xS ] = extract_HBF_network_parameters(xW,no,ni,nf,bias_neuron);

% the output
switch bias_neuron
    case 'with_bias'
        for j = 1 : size(X,2)
            for k = 1 : size(xC,2)
                h(k) = HBF_gaussian_activation_function(X(:,j),xC(:,k),xS(:,k));
            end
            y(j) = xW' * h' +  bias;
        end
    case 'no_bias'
        for j = 1 : size(X,2)
            for k = 1 : size(xC,2)
                h(k) = HBF_gaussian_activation_function(X(:,j),xC(:,k),xS(:,k));
            end
            y(j) = xW' * h';
        end
end