function predicted_labels = my_knn(train_data, train_labels, test_data, k)
     
    n = size(test_data, 2);
    
    % Initialize the predicted labels array
    predicted_labels = zeros(n, 1);
    
    % Loop over each test sample
    for i = 1:n
        % Compute the Euclidean distances between the test sample and all training samples
        distances = sqrt(sum((train_data - test_data(:, i)).^2, 1));
        
        % Sort the distances and get the indices of the k nearest neighbors
        [~, indices] = sort(distances);
        nearest_indices = indices(1:k);
        
        % Get the class labels of the k nearest neighbors
        nearest_labels = train_labels(nearest_indices);
        
        % Predict the label as the majority class label among the k nearest neighbors
        predicted_labels(i) = mode(nearest_labels);
        
        

    end
end