close all;
clear;

fileName = 'mnist.mat';

% Getting Data
[X_train, Y_train, X_test,Y_test] = get_data(fileName);
n = size(X_train,3);

n_classes = size(unique(Y_train), 1);
n_train= size(Y_train,1);
n_test= size(Y_test, 1);

% Set the number of clusters and maximum number of iterations
k = 30;

% Running kmeans
[Y_out, centroids] = my_kmeans(X_train,Y_train,X_test, k);

% Initializing Confusion Matrix
CM = zeros(n_classes, n_classes);

for i=1:n_test
    x = Y_out(i)+1;
    y = Y_test(i)+1;
    CM(x, y)=CM(x, y)+1;
end 

% basic accuracy (compares cluster assignmnet to label)
accuracy = (100*trace(CM))/n_test;
fprintf('trace = %d\n', trace(CM));
fprintf('K = %d\n', k);
fprintf('Accuracy: %1f\n', accuracy);
