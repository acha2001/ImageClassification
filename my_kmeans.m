function [Y_out, centroids] = my_kmeans(data, Y_train, X_test ,k)

    disp('Semi-Supervised KMeanns')

    X = data;
    %Y_out = zeros(size(Y_train));
    Y_out = zeros(1,k);
    max_iter = 100;
    

    % Initialize the cluster centroids randomly    
    centroids = X(:, randperm(size(X, 2), k));
    old_centroids = X(:, randperm(size(X, 2), k));
    
    % Loop over the iterations
    for iter = 1:max_iter
        
        % Assign each point to the nearest centroid
        distances = pdist2(X', centroids');
        
        [~, assignments] = min(distances, [], 2);
    
        % Update the centroids
        for i = 1:k
            centroids(:, i) = mean(X(:, assignments == i), 2);
        end

        % Check if the centroids have changed
        if iter > 1 && isequal(old_centroids, centroids)
            break; 
        end

        % Save the current assignments
        old_centroids = centroids;
    end

    % Find the closeset example to the each cluster and label that 
    K_assignment = zeros(k,1);
    for i = 1:k
        
        distances =  pdist2(X', centroids(:,i)');
        [~, min_distance] = min(distances);     
        K_assignment(i) = Y_train(min_distance);
    end

 % assign values to the test set

    n_test = size(X_test, 2);
    for i = 1:n_test
        d = pdist2(centroids', X_test(:,i)');
        [~, m] = min(d);
        Y_out(i) = K_assignment(m);  
    end
end
