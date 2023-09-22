close all;
clear;

fileName = 'mnist.mat';

% Getting Data
[X_train, Y_train, X_test,Y_test] = get_data(fileName);

% Getting various size variables 
n_classes = size(unique(Y_train), 1);
n_train= size(Y_train,1);
n_test= size(Y_test,1);

% MLP (multi layer perceptron = feedforward neural network with 1 hidden layer)

% Training model
Mdl_50 = fitcnet(X_train',Y_train',"LayerSizes", 50);
Mdl_100 = fitcnet(X_train',Y_train',"LayerSizes", 100);

% Initializing CM's
CM_50=zeros(n_classes,n_classes);
CM_100=zeros(n_classes,n_classes);

% Runing Test Data/Creating CM
[Y_out, ~] = predict(Mdl_50,X_test');

for i=1:n_test
    x = Y_test(i)+1;
    y = Y_out(i)+1;
    CM_50(x, y)=CM_50(x, y)+1;
end 

[Y_out, ~] = predict(Mdl_100,X_test');

for i=1:n_test
    x = Y_test(i)+1;
    y = Y_out(i)+1;
    CM_100(x, y)=CM_100(x, y)+1;
end 

% Calculate accuracy
accuracy_50 = (100*trace(CM_50))/n_test;
accuracy_100 = (100*trace(CM_100))/n_test;

disp(['Accuracy (50 Neurons): ' num2str(accuracy_50)]);
disp(['Accuracy (100 Neurons): ' num2str(accuracy_100)]);
