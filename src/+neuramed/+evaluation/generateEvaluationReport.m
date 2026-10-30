function report = generateEvaluationReport(yTrue, yPred, scores)
%GENERATEEVALUATIONREPORT Build a comprehensive single-call evaluation report.
%
%   REPORT = neuramed.evaluation.generateEvaluationReport(YTRUE, YPRED)
%   computes accuracy, confusion matrix, precision/recall,
%   sensitivity/specificity, and F1 in one call — the standard report
%   used across the toolkit (classical ML and deep learning alike) for
%   consistency.
%
%   REPORT = neuramed.evaluation.generateEvaluationReport(YTRUE, YPRED, SCORES)
%   additionally computes ROC/AUC when the problem is binary and SCORES
%   (posterior probability for the positive class, or an Nx2 prob matrix)
%   is supplied.
%
%   Example:
%       [pred, probs] = neuramed.models.predictModel(model, Xtest);
%       report = neuramed.evaluation.generateEvaluationReport(Ytest, pred, probs);
%
%   See also NEURAMED.EVALUATION.COMPUTEACCURACY, NEURAMED.EVALUATION.COMPUTEROCAUC

    report = struct();
    report.Accuracy = neuramed.evaluation.computeAccuracy(yTrue, yPred);
    [report.ConfusionMatrix, report.ClassNames] = neuramed.evaluation.computeConfusionMatrix(yTrue, yPred);
    report.PrecisionRecall = neuramed.evaluation.computePrecisionRecall(yTrue, yPred);
    report.SensitivitySpecificity = neuramed.evaluation.computeSpecificitySensitivity(yTrue, yPred);
    f1table = neuramed.evaluation.computeF1Score(yTrue, yPred);
    report.F1 = f1table;
    report.MacroF1 = f1table.Properties.UserData.MacroF1;

    report.AUC = NaN;
    report.ROC = [];
    if nargin >= 3 && ~isempty(scores) && numel(report.ClassNames) == 2
        if size(scores, 2) == 2
            posScores = scores(:, 2);
        else
            posScores = scores(:, 1);
        end
        try
            roc = neuramed.evaluation.computeROCAUC(yTrue, posScores);
            report.AUC = roc.AUC;
            report.ROC = roc;
        catch ME
            warning('neuramed:evaluation:generateEvaluationReport:ROCFailed', ...
                'ROC/AUC computation skipped: %s', ME.message);
        end
    end

    report.GeneratedAt = datetime('now');
end
