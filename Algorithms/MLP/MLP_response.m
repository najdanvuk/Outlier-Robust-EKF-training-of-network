function y = MLP_response(mlpnet,X)

% very very simple and non innovative...but if you realy want some fency and clever way for generating network
% response, you are wellcome.... ;)
% For example, take a look below....

numlay = size(mlpnet,2);

if numlay == 2;
    z1 = mlpnet(1).weights' * X + mlpnet(1).biases;         % from input to hidden
    h1 = tanh(z1);                                          % propagate z1 through tanh
    y = mlpnet(2).weights' * h1;                            % from hidden to output
else numlay == 3;
    z1 = mlpnet(1).weights' * X + mlpnet(1).biases;         % from input to first hidden
    h1 = tanh(z1);                                          % activate first hidden
    z2 = mlpnet(2).weights' * h1 + mlpnet(2).biases;        % from first hidden to second hidden
    h2 = tanh(z2);                                          % activate second hidden
    y = mlpnet(3).weights' * h2;                            % from second hidden to output
end


% 
% % 
% % numlay = size(mlpnet,2);
% % 
% % switch act_func
% %     case 'log_sig'
% %         for i = 2 : numlay-1
% %             W = mlpnet(i-1).weights;Z = mlpnet(i-1).z;
% %             Z = (1 + exp(-W' * Z)).^-1;mlpnet(i).z = Z;
% %         end
% %         W = mlpnet(numlay-1).weights;Z = mlpnet(numlay-1).z;
% %         Z = W' * Z;mlpnet(numlay).z = Z;
% %     case 'tan_sig'
% %         for i = 2 : numlay-1
% %             W = mlpnet(i-1).weights;Z = mlpnet(i-1).z;
% %             Z = (1 - exp(-2 * W' * Z))./(1 + exp(-2 * W' * Z));mlpnet(i).z = Z;
% %         end
% %         W = mlpnet(numlay-1).weights;Z = mlpnet(numlay-1).z;
% %         Z = W' * Z;mlpnet(numlay).z = Z;
% %         
% %     case 'tan_sig_2' % "Choi's" function: sigmoid in [-1,1] range
% %         for i = 2 : numlay-1
% %             W = mlpnet(i-1).weights;Z = mlpnet(i-1).z;
% %             Z = 2./(1 + exp(- W' * Z));mlpnet(i).z = Z;
% %         end
% %         W = mlpnet(numlay-1).weights;Z = mlpnet(numlay-1).z;
% %         Z = W' * Z;mlpnet(numlay).z = Z;       
% % end
% % return
