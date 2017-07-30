% % These functions are taken from S. Balasundaram et al. / Neural Networks 51 (2014) 67–79
% clc,clear,close all

inkr_train = 200;
inkr_test = 1000;
pct_of_outliers = 1;
offset_of_outliers = .2;

switch function_number
    %==========================================================================
    %==========================================================================
    % % 1. Function # 1
    case 1
        X1 = linspace(0,1,inkr_train);
        X2 = linspace(0,1,inkr_train);
        Xtrain = [ X1  ; X2 ];
        Ytrain = exp(Xtrain(1,:) .* sin(pi * Xtrain(2,:)));
        
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + offset_of_outliers .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        X1 = linspace(0,1,inkr_test);
        X2 = linspace(0,1,inkr_test);
        Xtest = [ X1  ; X2 ];
        Ytest = exp(Xtest(1,:) .* sin(pi * Xtest(2,:)));
        
        figure(313);
        plot(Ytrain,'r');hold on;
        plot(Ytest)
        %==========================================================================
        %==========================================================================
        % % 2. Function # 2
    case 2
        X1 = linspace(0,1,inkr_train);
        X2 = linspace(0,1,inkr_train);
        Xtrain = [ X1  ; X2 ];
        Ytrain = 1.9 * (1.35 + exp(Xtrain(1,:)).*sin(13*(Xtrain(1,:)-.6).^2) + exp(3*(Xtrain(2,:)-.5)) .* sin(4*pi*(Xtrain(2,:)-.9).^2))
        
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + offset_of_outliers .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        X1 = linspace(0,1,inkr_test);
        X2 = linspace(0,1,inkr_test);
        Xtest = [ X1  ; X2 ];
        
        Ytest = 1.9 * (1.35 + exp(Xtest(1,:)).*sin(13*(Xtest(1,:)-.6).^2) + exp(3*(Xtest(2,:)-.5)) .* sin(4*pi*(Xtest(2,:)-.9).^2))
        
        
        figure(313);
        plot(Ytrain,'r');hold on;
        plot(Ytest)
        %==========================================================================
        %==========================================================================
        % % 3. Function # 3
    case 3
        X1 = linspace(-2,2,inkr_train);
        X2 = linspace(-2,2,inkr_train);
        Xtrain = [ X1  ; X2 ];
        Ytrain = (1 + sin(2*Xtrain(1,:) + 3*Xtrain(2,:))) ./ (3.5 + sin(Xtrain(1,:)-Xtrain(2,:)))
        
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + offset_of_outliers .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        X1 = linspace(-1,1,inkr_test);
        X2 = linspace(-1,1,inkr_test);
        Xtest = [ X1  ; X2 ];
        
        Ytest = (1 + sin(2*Xtest(1,:) + 3*Xtest(2,:))) ./ (3.5 + sin(Xtest(1,:)-Xtest(2,:)))
        
        figure(313);
        plot(Ytrain,'r');hold on;
        plot(Ytest)
        % %==========================================================================
        % %==========================================================================
        % % % 4. Function # 4
    case 4
        X1 = linspace(0,100,inkr_train);
        X2 = linspace(40*pi,560*pi,inkr_train);
        X3 = linspace(0,1,inkr_train);
        X4 = linspace(1,11,inkr_train);
        Xtrain = [ X1  ; X2 ; X3 ; X4 ];
        
        Ytrain = atan((Xtrain(2,:).*Xtrain(3,:)-1./(Xtrain(2,:).*Xtrain(4,:))./Xtrain(1,:)))
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + offset_of_outliers .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        X1 = linspace(0,100,inkr_test);
        X2 = linspace(40*pi,560*pi,inkr_test);
        X3 = linspace(0,1,inkr_test);
        X4 = linspace(1,11,inkr_test);
        
        Xtest = [ X1  ; X2 ; X3 ; X4 ];
        
        Ytest = atan((Xtest(2,:).*Xtest(3,:)-1./(Xtest(2,:).*Xtest(4,:))./Xtest(1,:)))
        
        figure(313);
        plot(Ytrain,'r');hold on;
        plot(Ytest)
        %==========================================================================
        %==========================================================================
        % % 5. Function # 5
    case 5
        X1 = linspace(0,100,inkr_train);
        X2 = linspace(40*pi,560*pi,inkr_train);
        X3 = linspace(0,1,inkr_train);
        X4 = linspace(1,11,inkr_train);
        Xtrain = [ X1  ; X2 ; X3 ; X4 ];
        
        Ytrain = sqrt(Xtrain(1,:).^2 + (Xtrain(2,:).*Xtrain(3,:) - 1 ./(Xtrain(2,:).*Xtrain(4,:))).^2);
        
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + offset_of_outliers .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        X1 = linspace(0,100,inkr_test);
        X2 = linspace(40*pi,560*pi,inkr_test);
        X3 = linspace(0,1,inkr_test);
        X4 = linspace(1,11,inkr_test);
        
        Xtest = [ X1  ; X2 ; X3 ; X4 ];
        
        Ytest = sqrt(Xtest(1,:).^2 + (Xtest(2,:).*Xtest(3,:) - 1 ./(Xtest(2,:).*Xtest(4,:))).^2);
        
        figure(313);
        plot(Ytrain,'r');hold on;
        plot(Ytest)
        
        
        %==========================================================================
        %==========================================================================
        % % 5. Function # 6
    case 6
        X1 = linspace(0,1,inkr_train);
        X2 = linspace(0,1,inkr_train);
        X3 = linspace(0,1,inkr_train);
        X4 = linspace(0,1,inkr_train);
        X5 = linspace(0,1,inkr_train);
        Xtrain = [ X1  ; X2 ; X3 ; X4 ; X5];
        
        Ytrain = .79 + 1.27 * Xtrain(1,:) .* Xtrain(2,:) + 1.56 * Xtrain(1,:) .* Xtrain(4,:) + 3.42 * Xtrain(2,:).* Xtrain(5,:) + 2.06* Xtrain(3,:).* Xtrain(4,:).* Xtrain(5,:)
        
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + offset_of_outliers .* (2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        X1 = linspace(0,1,inkr_test);
        X2 = linspace(0,1,inkr_test);
        X3 = linspace(0,1,inkr_test);
        X4 = linspace(0,1,inkr_test);
        X5 = linspace(0,1,inkr_test);
        
        Xtest = [ X1  ; X2 ; X3 ; X4 ; X5 ];
        
        Ytest = .79 + 1.27 * Xtest(1,:) .* Xtest(2,:) + 1.56 * Xtest(1,:) .* Xtest(4,:) + 3.42 * Xtest(2,:).*Xtest(5,:) + 2.06* Xtest(3,:).* Xtest(4,:).* Xtest(5,:)
        
        figure(313);
        plot(Ytrain,'r');hold on;
        plot(Ytest)
end