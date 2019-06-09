function [ F1Macro_label ] = F1_macro_label( predict, true )
% Computing Macro-averaging F1 based label, for example
% dimension(predict) = instance * class
% predict = [-1, 1, 1; 1, 1, -1];true = [1, -1, 1; 1, 1, -1] 
% return 0.78
    % convert -1 to 0
    predict(predict == -1) = 0;
    true(true == -1) = 0;
    
    [~, num_class] = size(true);
    sum_tmp = 0;
    for i = 1: num_class
        TP = sum((true(:, i) == 1) & (predict(:, i) == 1));
        FP = sum((true(:, i) == 0) & (predict(:, i) == 1));
        TN = sum((true(:, i) == 0) & (predict(:, i) == 0));
        FN = sum((true(:, i) == 1) & (predict(:, i) == 0));
        
        if 2 * TP + FN + FP == 0
            continue;
        end
        F1 = 2 * TP / (2 * TP + FN + FP);
        sum_tmp = sum_tmp + F1;
    end 
    F1Macro_label = sum_tmp / num_class;
end

