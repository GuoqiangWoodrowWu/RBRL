function [ F1Example ] = F1_example_old( predict, true )
% Computing F1 based example, for example
% dimension(predict) = instance * class
% predict = [-1, 1, 1; 1, 1, -1];true = [1, -1, 1; 1, 1, -1] 
% return 3/4
    % convert -1 to 0
    predict(predict == -1) = 0;
    true(true == -1) = 0;
    
    PrecisionExample = Precision_example(predict, true);
    RecallExample = Recall_example(predict, true);
    % disp(['pe: ', num2str(PrecisionExample), ' re: ', num2str(RecallExample)]);
    F1Example = 2 * PrecisionExample * RecallExample / (PrecisionExample + RecallExample);
end

function [ PrecisionExample ] = Precision_example( predict, true )
% Computing Precision based example, for example
% predict = [0, 1, 1; 1, 1, 0];true = [1, 0, 1; 1, 1, 0] 
% return 3/4
    % convert -1 to 0
    predict(predict == -1) = 0;
    true(true == -1) = 0;

    [num_samples, ~] = size(true);
    both = predict & true;
    sum_tmp = 0;
    for i = 1: num_samples
        if sum(predict(i,:)) == 0
            continue;
        end
        sum_tmp = sum_tmp + sum(both(i, :)) / sum(predict(i,:));
    end
    PrecisionExample = sum_tmp / num_samples;
end

function [ RecallExample ] = Recall_example( predict, true )
% Computing Recall based example, for example
% predict = [0, 1, 1; 1, 1, 0];true = [1, 0, 1; 1, 1, 0] 
% return 3/4
    % convert -1 to 0
    predict(predict == -1) = 0;
    true(true == -1) = 0;
    
    [num_samples, ~] = size(true);
    both = predict & true;
    sum_tmp = 0;
    for i = 1: num_samples
        if sum(true(i,:)) == 0
            continue;
        end
        sum_tmp = sum_tmp + sum(both(i, :)) / sum(true(i,:));
    end
    RecallExample = sum_tmp / num_samples;
end