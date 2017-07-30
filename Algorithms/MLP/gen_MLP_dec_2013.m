function mlpnet = gen_MLP_dec_2013(lengths , numhidden,weightVar, biasVar)

lenX =lengths(1);
lenY = lengths(2);

% numhidden = [2 3 4] => three hidden layers with 2, 3 and 4 neurons
% respectively in each layer.
% initialize network parameters
if numel(numhidden) == 1
    mlpnet(1).weights =  sqrt(weightVar) * randn(lenX , numhidden);
    mlpnet(2).weights =  sqrt(weightVar) * randn(numhidden ,lenY);
    
    mlpnet(1).biases = sqrt(biasVar) .* rand(numhidden,1);
    
else numel(numhidden) == 2
    
    mlpnet(1).weights =  sqrt(weightVar) * randn(lenX , numhidden(1));
    mlpnet(2).weights =  sqrt(weightVar) * randn(numhidden(1),numhidden(2));
    mlpnet(3).weights =  sqrt(weightVar) * randn(numhidden(2),lenY);
    
    mlpnet(1).biases = sqrt(biasVar) .* rand(numhidden(1),1);
    mlpnet(2).biases = sqrt(biasVar) .* rand(numhidden(2),1);
end


% mlpnet(i).weights = sqrt(weightVar) * randn(numlay(i),numlay(i+1));%
% end
% for i = 1 : (size(numlay,2)),
%     mlpnet(i).z = zeros(1,1);
% end
% 
% if numel(numhidden) == 1
%     mlpnet(1).biases = sqrt(biasVar) .* randn(lenX,1);
% else numel(numhidden) == 2
%     for kk = 1 : numel(numlay)-2
%         mlpnet(kk).biases = sqrt(biasVar) .* randn(numlay(kk),1);
%     end
% end
% mlpnet(1).z = X;
% mlpnet(end).z = Y;
return
