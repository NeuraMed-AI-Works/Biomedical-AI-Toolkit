function cvResults = crossValidateModel(trainFcn, data, options)
%CROSSVALIDATEMODEL Run K-fold cross-validation for any NeuraMed training function.
%
%   CVRESULTS = neuramed.evaluation.crossValidateModel(@neuramed.models.trainRandomForest, data)
%   runs 5-fold stratified CV (via neuramed.dataset.crossValidationSplit),
%   trains TRAINFCN on each fold's training split, evaluates on the held
%   -out fold, and aggregates metrics.
%
%   Name-Value Options:
%       'K'       - number of folds (default 5)
%       'Seed'    - RNG seed (default 42)
%
%   Output CVRESULTS fields:
%       .PerFold          - struct array of per-fold evaluation reports
%       .MeanAccuracy / .StdAccuracy
%       .MeanF1 / .StdF1
%
%   Mathematical Notes:
%       CV_error = (1/K) sum_k Error(model_k, TestFold_k), with the
%       standard deviation across folds serving as an empirical estimate
%       of generalization-error variance — critical for small biomedical
%       cohorts where a single train/test split can be misleading.
%
%   Example:
%       cvResults = neuramed.evaluation.crossValidateModel(@neuramed.models.trainSVM, data, 'K', 10);
%       fprintf('Mean Accuracy: %.3f +/- %.3f\n', cvResults.MeanAccuracy, cvResults.StdAccuracy);
%
%   See also NEURAMED.DATASET.CROSSVALIDATIONSPLIT, NEURAMED.EVALUATION.GENERATEEVALUATIONREPORT

    arguments
        trainFcn (1,1) function_handle
        data
        options.K (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.K, 2)} = 5
        options.Seed (1,1) double = 42
    end

    folds = neuramed.dataset.crossValidationSplit(data, 'K', options.K, 'Seed', options.Seed);
    K = numel(folds);

    perFold = repmat(struct('Report', [], 'Model', []), 1, K);
    accs = zeros(K, 1);
    f1s = zeros(K, 1);

    for k = 1:K
        trainSet = folds(k).Train;
        testSet = folds(k).Test;

        model = trainFcn(trainSet.X, trainSet.Y);
        pred = neuramed.models.predictModel(model, testSet.X);
        report = neuramed.evaluation.generateEvaluationReport(testSet.Y, pred);

        perFold(k).Report = report;
        perFold(k).Model = model;
        accs(k) = report.Accuracy;
        f1s(k) = report.MacroF1;
    end

    cvResults = struct();
    cvResults.PerFold = perFold;
    cvResults.MeanAccuracy = mean(accs);
    cvResults.StdAccuracy = std(accs);
    cvResults.MeanF1 = mean(f1s);
    cvResults.StdF1 = std(f1s);
    cvResults.K = K;
end
