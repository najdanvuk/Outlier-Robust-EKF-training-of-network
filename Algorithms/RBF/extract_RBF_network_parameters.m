function [bias, Weights, Centers, Spreads] = extract_RBF_network_parameters(xW,no,ni,nf) 

bias = xW(1:no);

Weights = reshape(xW(no+1:no+nf*no),nf,no);

Centers = reshape(xW(no+nf*no+1:no+nf*no+nf*ni),ni,nf);

Spreads = xW(no+nf*no+nf*ni+1:end);
