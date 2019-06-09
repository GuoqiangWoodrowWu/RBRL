clear;
clc;

tic;
%% read the dataset
% the dataset for the kernel model has been normalized with 0 mean and 1 std.
load('emotions.mat');
% append another feature (all equals to 1) to the data
num_feature_origin = size(X_train, 2);
X_train(:, num_feature_origin + 1) = 1;
X_test(:, num_feature_origin + 1) = 1;

%% train the model in the train dataset and predict in the test dataset 
NIter = 2000;
% set the RBF kernel hyper-parameter
sigma = 1 / num_feature_origin; 
lambda1 = 1;
lambda2 = 0.01;
lambda3 = 0.1;
%% For RBF kernel model
[ A, obj ] = train_kernel_RBRL_APG( X_train, Y_train, lambda1, lambda2, lambda3, sigma, NIter);
[ pre_Label_test, pre_F_test ] = Kernel_Predict( X_train, X_test, A, sigma );
%% For linear model
%[ W, obj ] = train_linear_RBRL_APG( X_train, Y_train, lambda1, lambda2, lambda3, NIter );
%[ pre_Label_test, pre_F_test ] = Predict( X_test, W );

%% evaluate the performance of the model
[ test_HammingLoss, test_SubsetAccuracy, test_AccuracyExample, test_F1_Micro_Label, test_F1_Macro_Label, test_F1_Example, ...
    test_Ranking_Loss, test_Coverage, test_Average_Precision, test_One_Error, test_AUC_Macro_Label ] = Evaluation_Metrics( pre_Label_test, pre_F_test, Y_test );
plot(obj);

toc;