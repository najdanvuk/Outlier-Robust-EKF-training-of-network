function Ynew = RBF_response_dec_15(xW,X,no,ni,nf)

[bias, xW, xC, xS] = extract_RBF_network_parameters(xW,no,ni,nf);

invxS2 = xS.^-2;

sizX = size(X,2);

Y2 = zeros(sizX,nf);
Y2new=zeros(sizX,1);

for kk = 1 : nf
    Data = X' - repmat(xC(:,kk)',sizX,1);
    prob1 = sum(Data*invxS2(kk).*Data, 2);
    prob2 = exp(-0.5*prob1);
    Y2(:,kk) = prob2;
end

Ynew = Y2 * xW + bias;
Ynew = Ynew';