function [ W_alpha_1, obj ] = train_linear_RBRL_APG( X, Y, lambda1, lambda2, lambda3, NIter )
% Summary of this function: To train the linear RBRL by APG
% Input: size(X) = [n_instances, n_features] 
%        size(Y) = size(n_instances, n_labels)
%        Y \in {-1, +1}
% Output: size(W) = [n_features, n_labels]
% Written by Guoqiang Wu
    
    [num_instance, num_feature] = size(X);
    num_class = size(Y, 2);
    
    W_beta = zeros(num_feature, num_class);
    W_alpha_1 = zeros(num_feature, num_class);
    W_alpha_0 = zeros(num_feature, num_class);
    V = zeros(num_feature, num_class);
    
    j = 1;
    epsilon = 10^-6;
    lipschitz_1_rank = 0;
    if lambda2 ~= 0
        lipschitz_1_rank = calculate_lip_constant_ranking_loss(X, Y);
    end
    lipschitz_1 = sqrt(3 * (norm(X, 'fro')^2)^2 + ...
        3 * lambda1^2 + 3 * lambda2^2 * lipschitz_1_rank^2);
    t_1 = 1;
    while ((j <= 2 || abs(obj(j - 1) - obj(j - 2)) / obj(j - 2) > epsilon) && j < NIter)
        % Calculate the gradient
        [ V ] = Calculate_Gradient( X, Y, W_beta, lambda1, lambda2);
        
        W_alpha_0 = W_alpha_1;
        W_alpha_1 = W_beta - 1 / (lipschitz_1) * V; 
        
        if lambda3 ~= 0
            [u, s, v] = svd(W_alpha_1);
            [row, column] = size(s);
            s = s - lambda3 /lipschitz_1 * eye(row, column);
            s(s < 0) = 0;
            W_alpha_1 = u * s * v';
        end
        
        t_0 = t_1;
        t_1 = (1 + sqrt(1 + 4 * t_1^2)) / 2;
        W_beta = W_alpha_1 + (t_0 - 1) / t_1 * (W_alpha_1 - W_alpha_0);
        
        % Calculate the objective function value
        obj(j) = fValue( X, Y, W_alpha_1, lambda1, lambda2, lambda3 );
        j = j + 1;
    end
    % plot(obj);

end

function [ f_value ] = fValue( X, Y, W, lambda_1, lambda_2, lambda_3 )
    [num_instance, num_class] = size(Y);
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    temp = max(Z, I - Y .* (X * W));
    f_value = 0.5 * sum(sum(temp .* temp, 2)) + 0.5 * lambda_1 * norm(W, 'fro')^2;
    
    if lambda_2 ~= 0
        f_value_rank = calculate_fValue_ranking_loss( X, Y, W );
        f_value = f_value + lambda_2 * f_value_rank;
    end
    
    if lambda_3 ~= 0
        f_value = f_value + lambda_3 * sum(svd(W));
    end  
end

function [ V ] = Calculate_Gradient( X, Y, W, lambda_1, lambda_2)
    [num_instance, num_class] = size(Y);
    [~, num_feature] = size(X);
    
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    grad = X' * (-Y .* max(Z, I - Y .* (X * W)));
    
    V_rank = zeros(num_feature, num_class);
    if lambda_2 ~= 0
        V_rank = calculate_gradient_ranking_loss( X, Y, W );
    end  
    V = grad + lambda_2 * V_rank + lambda_1 * W;
end

function [ lip_constant ] = calculate_lip_constant_ranking_loss( X, Y )
    [n_samples, n_labels] = size(Y);
    A = zeros(n_labels, 1);
    B = zeros(n_labels, 1);
    for i = 1: n_samples
        num_positive = sum(Y(i,:) > 0); 
        num_negative = n_labels - num_positive;
        for j = 1: n_labels
            if Y(i,j) > 0
                A(j) = A(j) + num_negative;
                B(j) = B(j) + num_negative * norm(X(i,:))^4 / (num_positive^2 * num_negative^2);
            else
                A(j) = A(j) + num_positive;
                B(j) = B(j) + num_positive * norm(X(i,:))^4 / (num_positive^2 * num_negative^2);
            end
        end
    end
    lip_constant = sqrt(max(B .* A));
end

function [ W_gradient ] = calculate_gradient_ranking_loss( X, Y, W )
    [n_samples, n_features] = size(X);
    n_labels = size(Y, 2);
    W_gradient = zeros(n_features, n_labels);
    for i = 1: n_samples
        num_positive = sum(Y(i,:) > 0);
        num_negative = n_labels - num_positive;
        tmp_gradient = zeros(n_features, n_labels);
        for j = 1: n_labels
            if Y(i,j) > 0
                q_list = find(Y(i,:) < 0);
                for q = 1: length(q_list)
                    tmp_gradient(:, j) = tmp_gradient(:, j) + (-X(i,:)') * max(0, 2 - dot(W(:,j)-W(:,q_list(q)),X(i,:)));
                end
            else
                p_list = find(Y(i,:) > 0);
                for p = 1: length(p_list)
                    tmp_gradient(:, j) = tmp_gradient(:, j) + X(i,:)' * max(0, 2 - dot(W(:,p_list(p))-W(:,j),X(i,:)));
                end
            end
        end
        W_gradient = W_gradient + tmp_gradient /(num_positive * num_negative);
    end
end

function [ f_value ] = calculate_fValue_ranking_loss( X, Y, W )
    [n_samples, ~] = size(Y);
    f_value = 0;
    for i = 1: n_samples
        p_list = find(Y(i,:) > 0);
        q_list = find(Y(i,:) < 0);
        num_positive = length(p_list);
        num_negative = length(q_list);
        tmp_value = 0;
        for p = 1: num_positive
            for q = 1: num_negative
                tmp_value = tmp_value + max(0, 2 - dot(X(i,:), W(:,p_list(p)) - W(:,q_list(q))))^2;
            end
        end
        f_value = f_value + tmp_value/(num_positive * num_negative);
    end
    f_value = f_value / 2;
end