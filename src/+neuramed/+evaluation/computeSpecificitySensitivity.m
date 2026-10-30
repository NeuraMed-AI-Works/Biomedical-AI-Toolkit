function metrics = computeSpecificitySensitivity(yTrue, yPred)
%COMPUTESPECIFICITYSENSITIVITY Clinical sensitivity/specificity per class.
%
%   METRICS = neuramed.evaluation.computeSpecificitySensitivity(YTRUE, YPRED)
%   returns a table with per-class Sensitivity (= Recall) and Specificity,
%   the standard clinical-diagnostic metric pair.
%
%   Mathematical Definition (one-vs-rest per class c):
%       Sensitivity_c (True Positive Rate) = TP_c / (TP_c + FN_c)
%           "probability the test is positive given disease present"
%       Specificity_c (True Negative Rate) = TN_c / (TN_c + FP_c)
%           "probability the test is negative given disease absent"
%       These map directly to clinical screening-test interpretation and
%       are prioritized over raw accuracy in diagnostic contexts because
%       they separately quantify miss rate (1-Sensitivity) and
%       false-alarm rate (1-Specificity).
%
%   Example:
%       metrics = neuramed.evaluation.computeSpecificitySensitivity(Ytest, Ypred);
%
%   See also NEURAMED.EVALUATION.COMPUTEPRECISIONRECALL

    [C, classNames] = neuramed.evaluation.computeConfusionMatrix(yTrue, yPred);
    numClasses = numel(classNames);
    total = sum(C(:));
    EPSILON = 1e-12;

    sensitivity = zeros(numClasses, 1);
    specificity = zeros(numClasses, 1);

    for c = 1:numClasses
        TP = C(c, c);
        FN = sum(C(c, :)) - TP;
        FP = sum(C(:, c)) - TP;
        TN = total - TP - FN - FP;
        sensitivity(c) = TP / max(TP + FN, EPSILON);
        specificity(c) = TN / max(TN + FP, EPSILON);
    end

    metrics = table(classNames, sensitivity, specificity, ...
        'VariableNames', {'Class', 'Sensitivity', 'Specificity'});
end
