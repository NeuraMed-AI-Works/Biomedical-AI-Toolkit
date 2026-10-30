function acc = computeAccuracy(yTrue, yPred)
%COMPUTEACCURACY Overall classification accuracy.
%
%   ACC = neuramed.evaluation.computeAccuracy(YTRUE, YPRED)
%
%   Mathematical Definition:
%       Accuracy = (TP + TN) / (TP + TN + FP + FN) = (1/N) sum_n 1[yhat_n = y_n]
%
%   Note: Accuracy can be misleading under class imbalance (common in
%   disease screening, where the negative class dominates); prefer
%   Sensitivity/Specificity/F1/AUC for imbalanced clinical datasets.
%
%   See also NEURAMED.EVALUATION.COMPUTEF1SCORE, NEURAMED.EVALUATION.COMPUTEROCAUC

    yTrue = categorical(yTrue);
    yPred = categorical(yPred);
    acc = mean(yTrue == yPred);
end
