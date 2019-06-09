function [ AccuracyExample ] = Accuracy_example( predict, true )
% Computing Accuracy based example, for example: 
% dimension(predict) = num_instance * num_class
% predict = [-1, 1, 1; 1, 1, -1];true = [1, -1, 1; 1, 1, -1];
% return 2/3

    % convert -1 to 0
    predict(predict == -1) = 0;
    true(true == -1) = 0;
    
    [num_samples, ~] = size(true);
    both = double(predict & true);
    or = double(predict | true);
    sum_tmp = 0.0;
    for i = 1: num_samples
        sum_tmp = sum_tmp + sum(both(i, :)) / sum(or(i, :));
    end
    AccuracyExample = sum_tmp / num_samples;
end

