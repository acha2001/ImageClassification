function [cluster_indices, centroids] = my_kmeans_sl(data, labels, k)
   
    disp("Supervised Kmeans")
    
    X = data;
    max_iter = 100;
    cluster_indices = zeros(size(data, 2), 1);

    % Initialize the cluster centroids randomly
    centroids = X(:, randperm(size(X, 2), k));

    % Loop over the iterations
    for iter = 1:max_iter
        % Assign each point to the nearest centroid
        distances = pdist2(X', centroids');
        [~, assignments] = min(distances, [], 2);
    
        % Update the centroids
        for i = 1:k
            % Find the data points assigned to the current cluster
            cluster_data = X(:, assignments == i);
            % Compute the majority class label for the cluster
            cluster_labels = labels(assignments == i);
            majority_label = mode(cluster_labels);
            % Update the centroid using the mean of the assigned data points
            centroids(:, i) = mean(cluster_data, 2);
            % Assign the majority class label to the cluster indices
            cluster_indices(assignments == i) = majority_label;
        end
        
        % Check if the assignments have changed
        if iter > 1 && all(assignments == old_assignments)
            break;
        end

        % Save the current assignments
        old_assignments = assignments;
    end

    % Visualize the cluster centroids
    figure;
    for i = 1:k
        subplot(2, k/2, i);
        imagesc(reshape(centroids(:, i), [28, 28]));
        title(sprintf('Cluster %d', i));
    end
end