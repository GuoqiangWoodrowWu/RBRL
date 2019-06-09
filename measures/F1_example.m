function [ F1Example ] = F1_example( predict, true )
% Computing F1 based example, for example
% dimension(predict) = instance * class
% predict = [-1, 1, 1; 1, 1, -1];true = [1, -1, 1; 1, 1, -1] 
% return 3/4
    % convert -1 to 0
    predict(predict == -1) = 0;
    true(true == -1) = 0;
    
    [num_instance, ~] = size(predict);
    tmp_sum = 0;
    for i = 1: num_instance
        numerator = sum(and(predict(i, :), true(i, :)));
        denominator = sum(predict(i, :)) + sum(true(i, :));
        tmp_sum = tmp_sum + 2 * numerator / denominator;
    end
    F1Example = tmp_sum / num_instance;
end

