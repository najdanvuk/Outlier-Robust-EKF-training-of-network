function y = RBF_response(xW,X,no,ni,nf)

[bias, xW, xC, xS] = extract_RBF_network_parameters(xW,no,ni,nf);

% the output

for j = 1 : size(X,2)
    for k = 1 : size(xC,2)
        h(k) = gaussian_RBF(X(:,j),xC(:,k),xS(k));
    end
    y(j) = xW' * h' +  bias;
end