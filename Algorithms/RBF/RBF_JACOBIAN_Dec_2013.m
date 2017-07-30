function [G] = RBF_JACOBIAN_Dec_2013(X,C,mu,W,no)

nf = size(mu,2);

% =========================================================================
% jacobian wrt weights
for j = 1 : nf
    h(j) = gaussian_RBF(X,mu(:,j),C(j));
end

hg = no - 1;
H = h';
for t = 1 : hg
    H = blkdiag(H,h');
end

H = [eye(no) ; H];
% =========================================================================
% jacobian wrt centers
% W = w(:,2:end);
Jac = [];
for m = 1 : no
    Jmu = [];
    for k = 1 : nf
            J = W(k,m) * gaussian_RBF(X,mu(:,k),C(k)) * (X-mu(:,k)) / C(k)^2;
        Jmu = [Jmu; J];
    end
    Jac = [Jac Jmu];
end
Jmu = Jac; %reshape(Jac,ni*nf,no)
% =========================================================================
% jacobian wrt widths
Jac_b = [];
for k = 1 : no
        b = [];
        for j = 1 : nf
            bder = W(j,k)*(X - mu(:,j))'*(X - mu(:,j)) ./(C(j).^3)...
                *gaussian_RBF(X,mu(:,j),C(j));
            b = [b; bder] ;
        end
        Jac_b = [Jac_b b];
%     end
end
G = [H;Jmu;Jac_b];
% =========================================================================
return