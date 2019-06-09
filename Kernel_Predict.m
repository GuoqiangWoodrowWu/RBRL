function [ predict_Label, predict_F ] = Kernel_Predict( X_train, X_test, A, sigma )
% Predict for the RBF kernel model
% Input: size(X_train) = [n_instances_train, n_features] 
%        size(X_test) = size(n_instances_test, n_labels)
%        size(A) = [n_instance, n_labels]
%        sigma: RBF Kernel hyperparameter
% Output: size(predict_Label) = [n_instances_test, n_labels], 
%        predict_Label \in {-1, 1}
%        size(predict_F) = [n_instances_test, n_labels], 
%        predict_F \in R

    num_instance_test = size(X_test, 1);
    num_class = size(A, 2);
    
    num_instance_train = size(X_train, 1);
    K_test = zeros(num_instance_test, num_instance_train);
    for i = 1: num_instance_test
        for j = 1: num_instance_train
            K_test(i, j) = exp(-sigma*norm(X_train(j,:) - X_test(i,:), 2)^2);
            %K_test(i, j) = dot(X_train(j,:), X_test(i,:));
        end
    end
    predict_F = K_test * A;
    
    threshold = 0;
    predict_Label = double(predict_F > threshold);
    predict_Label(predict_Label < 1) = -1;
    
    for j = 1: num_instance_test
        if sum(predict_Label(j, :)) == -num_class
            max_column_index = find(predict_F(j, :) == max(predict_F(j, :)), 1);
            predict_Label(j, max_column_index) = 1;
        end
    end

end

