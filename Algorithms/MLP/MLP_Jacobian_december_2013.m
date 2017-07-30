function J = MLP_Jacobian_december_2013(lengths, hidden,mlp,X)

AF = 'tan_sig';
if numel(hidden) == 1;

    M = lengths(2);
    J = hidden ;
    K = lengths(1);
    
    z = mlp(1).weights' * X + mlp(1).biases;
    hh = tanh(z);
    
    %
    % Jacobian of network wrt input layer weights
    Jac_B = [];
    for i = 1 : M       %   counts dimensions of output vector
        J_B= [];
        for j = 1 : J    %   counts dimensions of first hidden vector
            new_jacb = [];
            for k = 1 : K       % counts dimensions of input vector
                jac_b = mlp(2).weights (j,i) * act_fun_derivative( hh(j) , AF ) * X(k);
                new_jacb = [new_jacb ; jac_b];
            end
            J_B = [J_B ; new_jacb];
        end
        Jac_B = [Jac_B  J_B];
    end
    J_B = Jac_B;
    
    % Jacobian wrt output layer weights
    JA = [];
    for i = 1 : M       %   counts dimensions of output vector
        JA = blkdiag(JA, hh);
    end
    J_A = JA;
    
    % Jacobian wrt bias in hidden layer
    Jbias = [];
    for i = 1 : M       %   counts dimensions of output vector
        jbias = [];
        for j = 1 : J    %   counts dimensions of first hidden vector
            jb = mlp(2).weights (j,i) * act_fun_derivative( hh(j) , AF );
            jbias = [jbias; jb];
        end
        Jbias = [Jbias jbias];
    end
    Jbias;
    % and finally, drums please.....
    % the Jacobian
    J = [J_B ; J_A ; Jbias];
    
else numel(hidden) == 2;
    K = lengths(1);
    M = lengths(2);
    J = hidden(1);
    N = hidden(2);

    
    z1 = mlp(1).weights' * X + mlp(1).biases;
    h1 = tanh(z1);
    
    z2 = mlp(2).weights' * h1 + mlp(2).biases;
    h2 = tanh(z2);
    
    y = mlp(3).weights' *  h2;
    
    % Jacobian of network wrt input layer weights
    Jac_B = [];
    for i = 1 : M       %   counts dimensions of output vector
        J_B= [];
        for j = 1 : N    %   counts dimensions of second hidden vector
            new_jacb = [];
            for k = 1 : J       % counts dimensions of first hidden vector
                jac_b = mlp(3).weights (j,i) * act_fun_derivative( h2(j) , AF ) * h1(k);
                new_jacb = [new_jacb ; jac_b];
            end
            J_B = [J_B ; new_jacb];
        end
        Jac_B = [Jac_B  J_B];
    end
    J_B = Jac_B;
    
    % Jacobian wrt output layer weights
    JA = [];
    for i = 1 : M       %   counts dimensions of output vector
        JA = blkdiag(JA, h2);
    end
    J_A = JA;
    
    % Jacobian wrt bias in hidden layer
    Jbias = [];
    for i = 1 : M       %   counts dimensions of output vector
        jbias = [];
        for j = 1 : N    %   counts dimensions of second hidden vector
            jb = mlp(3).weights (j,i) * act_fun_derivative( h2(j) , AF );
            jbias = [jbias; jb];
        end
        Jbias = [Jbias jbias];
    end
    Jbias;
    
    J_C = [];
    for m = 1 : M
        Jacobian_cc = [];
        for l = 1 : J
            Jacobian_c = [];
            for n = 1 : K
                Jac_c = 0;
                for k = 1 : N
                    jac_c = mlp(3).weights(k,m) * act_fun_derivative(h2(k),AF) * mlp(2).weights(l,k) *...
                        act_fun_derivative(h1(l),AF) * X(n);
                    Jac_c = Jac_c + jac_c;
                end
                Jacobian_c = [Jacobian_c;Jac_c];
            end
            Jacobian_cc = [Jacobian_cc; Jacobian_c];
        end
        J_C = [J_C Jacobian_cc];
    end
    J_C;
    %     wrt bias neuron in the first hidden layer
    J_c = [];
    for m = 1 : M
        Jacobian_cc = [];
        for l = 1 : J
            Jacobian_c = [];
                Jac_c = 0;
                for k = 1 : N
                    jac_c = mlp(3).weights(k,m) * act_fun_derivative(h2(k),AF) * mlp(2).weights(l,k) *...
                        act_fun_derivative(h1(l),AF);
                    Jac_c = Jac_c + jac_c;
                end
                Jacobian_c = [Jacobian_c;Jac_c];
            Jacobian_cc = [Jacobian_cc; Jacobian_c];
        end
        J_c = [J_c Jacobian_cc];
    end
    % and finally, drums please.....
    % the Jacobian
    J = [J_C;J_B ; J_A ; Jbias;J_c];
end



% In the state vector, in case of one layered network, we firstly set input weights, then output, and
% finally the bias in hidden layer
% J_check = [JA_check ; J_B_check ; Jbias_check]

% In the case of two layered network, the Jacobian is:
% J = [JC;JB;JA;Jc;Jb]
% that is, input to first hidden, from first hidden to second hidden and
% finally, from second hidden to output