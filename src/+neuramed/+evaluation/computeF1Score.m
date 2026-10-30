function metrics = computeF1Score(yTrue, yPred)
%COMPUTEF1SCORE Per-class and macro-averaged F1 score.
%
%   METRICS = neuramed.evaluation.computeF1Score(YTRUE, YPRED)
%
%   Mathematical Definition:
%       F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)
%       i.e. the harmonic mean of precision and recall, which penalizes
%       models that trade one off heavily against the other (unlike the
%       arithmetic mean). Macro-F1 = mean_c(F1_c) is a standard summary
%       metric for imbalanced multi-class clinical classification.
%
%   Example:
%       metrics = neuramed.evaluation.computeF1Score(Ytest, Ypred);
%
%   See also NEURAMED.EVALUATION.COMPUTEPRECISIONRECALL

    pr = neuramed.evaluation.computePrecisionRecall(yTrue, yPred);
    EPSILON = 1e-12;
    f1 = 2 * (pr.Precision .* pr.Recall) ./ max(pr.Precision + pr.Recall, EPSILON);

    metrics = table(pr.Class, pr.Precision, pr.Recall, f1, ...
        'VariableNames', {'Class', 'Precision', 'Recall', 'F1Score'});
    metrics.Properties.UserData = struct('MacroF1', mean(f1));
end
