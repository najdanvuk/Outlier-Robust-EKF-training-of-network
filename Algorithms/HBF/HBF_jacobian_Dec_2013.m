function J = HBF_jacobian_Dec_2013(X,W,mu,xC,no,bias)
% Input:
% X data
% W - matrix of weights (hidden x output)
% mu - centers (dimension x number of functions)
% xC - witdhs (dimension x number of functions); pay attention that xC is
% defined as vector.

switch bias
    
    case 'no_bias'
        
        nf = size(mu,2);
        
        % % =========================================================================
        % jacobian wrt weights
        for j = 1 : nf
            h(j) = HBF_gaussian_activation_function(X, mu(:,j), xC(:,j));
        end
        H = h';
        for t = 1 : (no - 1)
            H = blkdiag(H,h');
        end
        % % =========================================================================
        % jacobian wrt centers
        % jacobian wrt centers
        % W = w(:,2:end);
        Jac_mu = [];
        for m = 1 : no
            Jmu_mu = [];
            for k = 1 : nf
                jac = W(k,m) * HBF_gaussian_activation_function(X,mu(:,k),xC(:,k)) .* (X-mu(:,k)) ./ xC(:,k);
                Jmu_mu = [Jmu_mu; jac];
            end
            Jac_mu = [Jac_mu Jmu_mu];
        end
        % % =========================================================================
        % % % jacobian wrt widths
        JacS=[];
        for m = 1 : no
            JS = [];
            for k = 1 : nf
                jac = .5 * W(k,m) * HBF_gaussian_activation_function(X,mu(:,k),xC(:,k)) ...
                    .* (X-mu(:,k)).*(X-mu(:,k)) ./ xC(:,k).^2;
                JS = [JS; jac];
            end
            JacS = [JacS JS];
        end
        % % =========================================================================
        J = [ H ; Jac_mu ; JacS ] ;
        
    case 'with_bias'
        
         nf = size(mu,2);
        % % =========================================================================
        % jacobian wrt bias weights
        Jbias = eye(no);
        % % =========================================================================
        % jacobian wrt weights
        for j = 1 : nf
            h(j) = HBF_gaussian_activation_function(X, mu(:,j), xC(:,j));
        end
        H = h';
        for t = 1 : (no - 1)
            H = blkdiag(H,h');
        end
        % % =========================================================================
        % jacobian wrt centers
        % jacobian wrt centers
        % W = w(:,2:end);
        Jac_mu = [];
        for m = 1 : no
            Jmu_mu = [];
            for k = 1 : nf
                jac = W(k,m) * HBF_gaussian_activation_function(X,mu(:,k),xC(:,k)) .* (X-mu(:,k)) ./ xC(:,k);
                Jmu_mu = [Jmu_mu; jac];
            end
            Jac_mu = [Jac_mu Jmu_mu];
        end
        % % =========================================================================
        % % % jacobian wrt widths
        JacS=[];
        for m = 1 : no
            JS = [];
            for k = 1 : nf
                jac = .5 * W(k,m) * HBF_gaussian_activation_function(X,mu(:,k),xC(:,k)) ...
                    .* (X-mu(:,k)).*(X-mu(:,k)) ./ xC(:,k).^2;
                JS = [JS; jac];
            end
            JacS = [JacS JS];
        end
        % % =========================================================================
        J = [ Jbias ; H ; Jac_mu ; JacS ] ;
end