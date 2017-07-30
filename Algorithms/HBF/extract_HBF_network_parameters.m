function [ bias , Weights, Centers, Spreads ] = extract_HBF_network_parameters(xW,no,ni,nf,bias)

switch bias
    case 'with_bias'
        
        bias = xW(1:no);
        
        Weights = reshape(xW(no+1:no+nf*no),nf,no);
        
        Centers = reshape(xW(no+nf*no+1:no+nf*no+nf*ni),ni,nf);
        
        Spreads = reshape(xW(no+nf*no+nf*ni+1:end),ni,nf);
        
    case 'no_bias'
        
        Weights = reshape(xW(1:nf*no),nf,no);
        
        Centers = reshape(xW(nf*no+1:nf*no+nf*ni),ni,nf);
        
        Spreads = reshape(xW(nf*no+nf*ni+1:end),ni,nf);
        
        bias = [];
end