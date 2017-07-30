function mlp = extract_MLP_parameters_from_EKF(xW,hidden_layers,lengths)
% xW - state vector
rowX = lengths(1);
rowY = lengths(2);

% extract weights

IND = [];
layers = [rowX hidden_layers rowY];
for kk = 1 : (numel(layers)-1)
    in = layers(kk);
    out = layers(kk+1);
    t = in * out;
    IND = [IND t];
    if numel(IND) == 1
        index = 1 : t;
    elseif numel(IND) == 2
        index = IND(kk-1)+1 : IND(kk-1) + IND(kk);
    elseif numel(IND) == 3
        index = IND(kk-2) + IND(kk-1) + 1 : IND(kk-2) + IND(kk-1) + IND(kk);
    end
    mlp(kk).weights = reshape(xW(index),in,out);
end

xbias = xW(index(end)+1:end);

if numel(hidden_layers) == 1;
    mlp(1).biases = xbias;
else numel(hidden_layers) == 2;
    mlp(1).biases = xbias(1 : hidden_layers(1));
    mlp(2).biases = xbias(1 + hidden_layers(1) : end);
end