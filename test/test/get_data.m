function [my_X_train, my_Y_train, X_test,Y_test] = get_data(fileName)

% Load the data
mnist = load(fileName);

% Reshape training data into a 2D matrix
train_data = mnist.training.images;

n_train = size(train_data, 3);
train_data = train_data(:, :, 1:n_train); 
Y_train = mnist.training.labels;
tmp_X_train = reshape(train_data, [784, n_train]);

% get the first 500 examples of each class

nclasses = unique(Y_train);
nclasses = numel(nclasses);

my_X_train = zeros(28*28, 5000);
my_Y_train = zeros(5000,1);


for i = 1:nclasses
    class = i-1;
    indexes = find(Y_train == class, 500);
    tmpMat = tmp_X_train(:,indexes);
    my_X_train(:,(class*500)+1:(i)*500) = tmpMat;
    my_Y_train((class*500)+1:(i)*500) = Y_train(indexes);
end


% Reshap test data 
test_data = mnist.test.images;
n_test = size(test_data, 3);
test_data = test_data(:, :, 1:n_test);
X_test = reshape(test_data, [784, 10000]);
Y_test = mnist.test.labels;

end

