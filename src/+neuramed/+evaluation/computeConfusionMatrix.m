function [C, classNames] = computeConfusionMatrix(yTrue, yPred)
%COMPUTECONFUSIONMATRIX Compute the confusion matrix for classification results.
%
%   [C, CLASSNAMES] = neuramed.evaluation.computeConfusionMatrix(YTRUE, YPRED)
%   returns C(i,j) = count of observations with true class i predicted as
%   class j, over the sorted union of classes in YTRUE and YPRED.
%
%   Mathematical Definition:
%       C_{i,j} = sum_n 1[y_true_n = class_i] * 1[y_pred_n = class_j]
%       For binary classification with positive class as class 2:
%           TN = C(1,1), FP = C(1,2), FN = C(2,1), TP = C(2,2)
%
%   Example:
%       [C, names] = neuramed.evaluation.computeConfusionMatrix(Ytest, Ypred);
%
%   See also NEURAMED.EVALUATION.GENERATEEVALUATIONREPORT

    yTrue = categorical(yTrue);
    yPred = categorical(yPred);
    classNames = union(categories(yTrue), categories(yPred));
    yTrue = categorical(yTrue, classNames);
    yPred = categorical(yPred, classNames);

    C = confusionmat(yTrue, yPred);
end
