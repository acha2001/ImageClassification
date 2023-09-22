close all;
clear;

fileName = 'mnist.mat';

% Getting Data
[X_train, Y_train, X_test,Y_test] = get_data(fileName);


% getting various size variables 
n_classes = size(unique(Y_train), 1);
n_train= size(Y_train,1);
n_test= size(Y_test,1);

% Set the number of clusters and maximum number of iterations
k = 10;

% Calling KNN fucntion
Y_out = my_knn(X_train, Y_train, X_test, k); 

% Initializing Confusion Matrix
CM = zeros(n_classes, n_classes);

for i=1:n_test
    x = Y_test(i)+1;
    y = Y_out(i)+1;
    CM(x, y)=CM(x, y)+1;
end 

cm_acc = (100*trace(CM))/n_test;

fprintf('K = %d\n', k);
fprintf('Accuracy: %1f\n', cm_acc);




