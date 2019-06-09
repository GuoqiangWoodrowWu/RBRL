function [ A_alpha_1, obj ] = train_kernel_RBRL_APG( X, Y, lambda_1, lambda_2, lambda_3, sigma, NIter )
% Summary of this function: to train the kernel RBRL by APG
% Input: size(X) = [n_instances, n_features] 
%        size(Y) = size(n_instances, n_labels)
%        Y \in {-1, +1}
% Output: size(A) = [n_instances, n_labels]
% Written by Guoqiang Wu

    
    [num_instance, num_feature] = size(X);
    num_class = size(Y, 2);
    
    A_alpha_0 = zeros(num_instance, num_class);
    A_alpha_1 = zeros(num_instance, num_class);
    A_beta = zeros(num_instance, num_class);
    V = zeros(num_instance, num_class);
    
    % Calculate Kernel matrix
    % rbf kernel
    K = zeros(num_instance, num_instance);
    for i = 1: num_instance
        for j = 1: num_instance
            K(i, j) = exp(-sigma*norm(X(i,:) - X(j,:), 2)^2);
        end
    end
    
    j = 1;
    epsilon = 10^-6;
    lipschitz_rank = 0;
    if lambda_2 ~= 0
        lipschitz_rank = calculate_lip_constant_ranking_loss(K, Y);
    end
    lipschitz_1 = sqrt(3 * (norm(K, 'fro')^2)^2 + 3 * lambda_1^2 * (norm(K, 'fro')^2)... 
        + 3 * lambda_2^2 * lipschitz_rank^2);
    t_1 = 1;
    while ((j <= 2 || abs(obj(j - 1) - obj(j - 2)) / obj(j - 2) > epsilon) && j < NIter)
        % Calculate the gradient
        [ V ] = Calculate_Gradient( K, Y, A_beta, lambda_1, lambda_2 );
        A_alpha_0 = A_alpha_1;
        A_alpha_1 = A_beta - 1 / (lipschitz_1) * V; 
        
        if lambda_3 ~= 0
            [u, s, v] = svd(A_alpha_1);
            [row, column] = size(s);
            s = s - lambda_3 /lipschitz_1 * eye(row, column);
            s(s < 0) = 0;
            A_alpha_1 = u * s * v';
        end
        
        t_0 = t_1;
        t_1 = (1 + sqrt(1 + 4 * t_1^2)) / 2;
        A_beta = A_alpha_1 + (t_0 - 1) / t_1 * (A_alpha_1 - A_alpha_0);
        % Calculate the objective function value
        obj(j) = fValue( Y, A_alpha_1, lambda_1, lambda_2, lambda_3, K );
        j = j + 1;
    end
    % plot(obj);
end

function [ f_value ] = fValue( Y, A, lambda_1, lambda_2, lambda_3, K)
    [num_instance, num_class] = size(Y);
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    temp = max(Z, I - Y .* (K * A));
    f_value = 0.5 * sum(sum(temp .* temp, 2)) + 0.5 * lambda_1 * trace(A' * K * A);
    
    if lambda_2 ~= 0
        f_value_rank = calculate_fValue_ranking_loss(K, Y, A);
        f_value = f_value + lambda_2 * f_value_rank;
    end
    
    if lambda_3 ~= 0
        f_value = f_value + lambda_3 * sum(svd(A));
    end
end

function [ V ] = Calculate_Gradient( K, Y, A, lambda_1, lambda_2 )
    [num_instance, num_class] = size(Y);
    
    I = ones(num_instance, num_class);
    Z = zeros(num_instance, num_class);
    grad = K' * (-Y .* max(Z, I - Y .* (K * A)));
    
    V_rank = zeros(num_instance, num_class);
    if lambda_2 ~= 0
        V_rank = calculate_gradient_ranking_loss(K, Y, A);
    end
    
    V = grad + lambda_1 * (K * A) + lambda_2 * V_rank;
    
end

function [ lip_constant ] = calculate_lip_constant_ranking_loss( K, Y )
    [n_samples, n_labels] = size(Y);
    A = zeros(n_labels, 1);
    B = zeros(n_labels, 1);
    for i = 1: n_samples
        num_positive = sum(Y(i,:) > 0); 
        num_negative = n_labels - num_positive;
        for j = 1: n_labels
            if Y(i,j) > 0
                A(j) = A(j) + num_negative;
                B(j) = B(j) + num_negative * norm(K(i,:))^4 / (num_positive^2 * num_negative^2);
            else
                A(j) = A(j) + num_positive;
                B(j) = B(j) + num_positive * norm(K(i,:))^4 / (num_positive^2 * num_negative^2);
            end
        end
    end
    lip_constant = sqrt(max(B .* A));
end

function [ A_gradient ] = calculate_gradient_ranking_loss( K, Y, A )
    [n_samples, n_labels] = size(Y);
    A_gradient = zeros(n_samples, n_labels);
    for i = 1: n_samples
        num_positive = sum(Y(i,:) > 0);
        num_negative = n_labels - num_positive;
        tmp_gradient = zeros(n_samples, n_labels);
        for j = 1: n_labels
            if Y(i,j) > 0
                q_list = find(Y(i,:) < 0);
                for q = 1: length(q_list)
                    tmp_gradient(:, j) = tmp_gradient(:, j) + (-K(:,i)) * max(0, 2 - dot(A(:,j)-A(:,q_list(q)),K(:,i)));
                end
            else
                p_list = find(Y(i,:) > 0);
                for p = 1: length(p_list)
                    tmp_gradient(:, j) = tmp_gradient(:, j) + K(:,i) * max(0, 2 - dot(A(:,p_list(p))-A(:,j),K(:,i)));
                end
            end
        end
        A_gradient = A_gradient + tmp_gradient /(num_positive * num_negative);
    end
end

function [ f_value ] = calculate_fValue_ranking_loss( K, Y, A )
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
                tmp_value = tmp_value + max(0, 2 - dot(K(:,i), A(:,p_list(p)) - A(:,q_list(q))))^2;
            end
        end
        f_value = f_value + tmp_value/(num_positive * num_negative);
    end
    f_value = f_value / 2;
end