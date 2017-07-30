function Ynew = HBF_response_dec_2015(xW,X,no,ni,nf,bias_neuron)

[ bias , xW, xC, xS ] = extract_HBF_network_parameters(xW,no,ni,nf,bias_neuron);

sizX = size(X,2);

Y2 = zeros(sizX,nf);
% Y2new=zeros(sizX,1);

for kk = 1 : nf
    invP = diag(xS(:,kk).^-1);
    Data = X' - repmat(xC(:,kk)',sizX,1);
    prob1 = sum(Data*invP.*Data, 2);
    prob2 = exp(-0.5*prob1);
    Y2(:,kk) = prob2;
end

switch bias_neuron
    case 'with_bias'
        Ynew = Y2 * xW + bias;
    case 'no_bias'
        Ynew = Y2 * xW;
end
Ynew = Ynew';


% the output
% % switch bias_neuron
% %     case 'with_bias'
% %         for j = 1 : size(X,2)
% %             for k = 1 : size(xC,2)
% %                 h(k) = HBF_gaussian_activation_function(X(:,j),xC(:,k),xS(:,k));
% %             end
% %             y(j) = xW' * h' +  bias;
% %         end
% %     case 'no_bias'
% %         for j = 1 : size(X,2)
% %             for k = 1 : size(xC,2)
% %                 h(k) = HBF_gaussian_activation_function(X(:,j),xC(:,k),xS(:,k));
% %             end
% %             y(j) = xW' * h';
% %         end
% % end
