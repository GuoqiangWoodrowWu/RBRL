function [ F1Micro_label ] = F1_micro_label( predict, true )
% Computing Micro-averaging F1 based label, for example
% dimension(predict) = instance * class
% predict = [-1, 1, 1; 1, 1, -1];true = [1, -1, 1; 1, 1, -1] 
% return 0.75
    % convert -1 to 0
    predict(predict == -1) = 0;
    true(true == -1) = 0;    

    [~, num_class] = size(true);
    TP_sum = 0;
    FP_sum = 0;
    TN_sum = 0;
    FN_sum = 0;
    for i = 1: num_class
        TP = sum((true(:, i) == 1) & (predict(:, i) == 1));
        FP = sum((true(:, i) == 0) & (predict(:, i) == 1));
        TN = sum((true(:, i) == 0) & (predict(:, i) == 0));
        FN = sum((true(:, i) == 1) & (predict(:, i) == 0));
        TP_sum = TP_sum + TP;
        FP_sum = FP_sum + FP;
        TN_sum = TN_sum + TN;
        FN_sum = FN_sum + FN;
    end
    F1Micro_label = 2 * TP_sum / ( 2 * TP_sum + FN_sum + FP_sum);
end

