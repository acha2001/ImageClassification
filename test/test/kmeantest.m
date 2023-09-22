disp('Non Supervised KMeanns')
clc;
clear;

fileName = 'mnist.mat';

[X_train, Y_train, X_test,Y_test] = get_data(fileName);
k = 10;

    X = X_train;
    max_iter = 100;
    
    cluster_indices = zeros(size(X, 2),1);
    Y_out = zeros(size(X_train,2), 1);

    % Initialize the cluster centroids randomly
    centroids = X(:, randperm(size(X, 2), k));

    % Loop over the iterations
    for iter = 1:max_iter
        
        % Assign each point to the nearest centroid
        distances = pdist2(X', centroids');
        [~, assignments] = min(distances, [], 2);
    
        % Check if the assignments have changed
        if iter > 1 && all(assignments == old_assignments)
            break; 
        end
    
        % Update the centroids
        for i = 1:k
            centroids(:, i) = mean(X(:, assignments == i), 2);
        end

        % Save the current assignments
        old_assignments = assignments;
    end

    % Assign each point to its cluster index
    for i = 1:k
        cluster_indices(assignments == i) = i;
    end

    % Find the closeset example to the each cluster and label that 
    
    for i = 1:k
        
        index = find(cluster_indices == i);
        distances =  pdist2(X', centroids(:, i)');
        [~, min_distance_idx] = min(distances);
        
        Y_out(index) = Y_train(min_distance_idx);

        %cluster_indices = i;
    end

    % Visualize the cluster centroids
    figure;
    for i = 1:k
        subplot(2, k/2, i);
        imagesc(reshape(centroids(:, i), [28, 28]));
        title(sprintf('Cluster %d', i));
    end