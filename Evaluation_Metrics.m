function [ HammingLoss, SubsetAccuracy, AccuracyExample, F1_Micro_Label, F1_Macro_Label, F1_Example, Ranking_Loss, Coverage, Average_Precision, One_Error, AUC_Macro_Label ] = Evaluation_Metrics( pre_Label, pre_F, Y )
%UNTITLED4 Evaluate the model for many metrics
%   Detailed explanation goes here

    cd('./measures');
    HammingLoss = Hamming_loss(pre_Label, Y);
    SubsetAccuracy = Subset_accuracy(pre_Label, Y);
    AccuracyExample = Accuracy_example(pre_Label, Y);
    F1_Micro_Label = F1_micro_label(pre_Label, Y);
    F1_Macro_Label = F1_macro_label(pre_Label, Y);
    F1_Example = F1_example(pre_Label, Y);
    Ranking_Loss = Ranking_loss(pre_F, Y);
    Coverage = coverage_new(pre_F, Y);
    Average_Precision = Average_precision(pre_F, Y);
    One_Error = One_error(pre_F, Y);
    AUC_Macro_Label = AUC_macro_label(pre_F, Y);
    cd('../');
end
