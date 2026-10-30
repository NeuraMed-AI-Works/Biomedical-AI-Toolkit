function metrics = computePrecisionRecall(yTrue, yPred, options)
%COMPUTEPRECISIONRECALL Per-class and averaged precision/recall.
%
%   METRICS = neuramed.evaluation.computePrecisionRecall(YTRUE, YPRED)
%   returns a table of per-class Precision/Recall plus macro/micro
%   averages.
%
%   Name-Value Options:
%       'PositiveClass' - for binary problems, which class is "positive"
%                          (default: second category alphabetically)
%
%   Mathematical Definition:
%       Precision_c = TP_c / (TP_c + FP_c)   ("of predicted c, how many were right")
%       Recall_c    = TP_c / (TP_c + FN_c)   ("of actual c, how many were found")
%       Macro-average: mean over classes (treats each class equally,
%       robust to class imbalance in reporting).
%       Micro-average: pools all TP/FP/FN globally before computing the
%       ratio (dominated by majority class).
%
%   Example:
%       metrics = neuramed.evaluation.computePrecisionRecall(Ytest, Ypred);
%
%   See also NEURAMED.EVALUATION.COMPUTEF1SCORE, NEURAMED.EVALUATION.COMPUTECONFUSIONMATRIX

    arguments
        yTrue
        yPred
        options.PositiveClass = []
    end

    [C, classNames] = neuramed.evaluation.computeConfusionMatrix(yTrue, yPred);
    numClasses = numel(classNames);

    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    EPSILON = 1e-12;

    for c = 1:numClasses
        TP = C(c, c);
        FP = sum(C(:, c)) - TP;
        FN = sum(C(c, :)) - TP;
        precision(c) = TP / max(TP + FP, EPSILON);
        recall(c) = TP / max(TP + FN, EPSILON);
    end

    metrics = table(classNames, precision, recall, ...
        'VariableNames', {'Class', 'Precision', 'Recall'});
    metrics.Properties.UserData = struct( ...
        'MacroPrecision', mean(precision), 'MacroRecall', mean(recall), ...
        'MicroPrecision', sum(diag(C)) / max(sum(C(:)), EPSILON), ...
        'MicroRecall', sum(diag(C)) / max(sum(C(:)), EPSILON));
end
