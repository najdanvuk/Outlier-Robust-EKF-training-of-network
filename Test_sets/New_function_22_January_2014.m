% clc,clear,close all

pct_of_outliers = .4;
offset_of_outliers = .05;

switch function_case
    
    case 1
        
        Xtrain = linspace(-8,12,50);
        Ytrain = (Xtrain-2) .* (2*Xtrain - 1) ./ (1 + Xtrain.^2);
  
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + sqrt(offset_of_outliers) .* randn(size(Ytrain,1),number_of_outliers) ;%(2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        Xtest = linspace(-8,12,50);
        Ytest = (Xtest-2) .* (2*Xtest - 1) ./ (1 + Xtest.^2);
    
    case 2
        Xtrain = linspace(-1,1,51);
        for kk = 2 : numel(Xtrain)
            Ytrain(kk) = sin(5*Xtrain(kk)) * acos(Xtrain(kk)) * cos(3*Xtrain(kk-1));
        end
        Ytrain(1) = [];
        Xtrain(1) = [];
        
        
        % add outliers
        number_of_outliers = round(pct_of_outliers * size(Ytrain,2));
        
        index_of_outliers = randi([1 size(Xtrain,2)],1,number_of_outliers);
        
        Youtlier  = Ytrain(:,index_of_outliers) + sqrt(offset_of_outliers) .* randn(size(Ytrain,1),number_of_outliers) ;%(2*rand(size(Ytrain,1),number_of_outliers) - ones(size(Ytrain,1),number_of_outliers));
        Xoutlier = Ytrain(:,index_of_outliers);
        
        Ytrain(:,index_of_outliers) = Youtlier;
        
        
        Xtest = linspace(-1,1,51);
        for kk = 2 : numel(Xtest)
            Ytest(kk) = sin(5*Xtest(kk)) * acos(Xtest(kk)) * cos(3*Xtest(kk-1));
        end
        Ytest(1) = [];
        Xtest(1) = [];
 
end
figure(3333);
plot(Ytrain,'b'),hold on,plot(Ytest,'r');