function tests = test_evaluation_module
%TEST_EVALUATION_MODULE Unit tests for neuramed.evaluation.*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root, 'startup.m'));
end

function test_computeAccuracy_perfectPredictions(testCase)
    Y = categorical({'A';'B';'A';'B'});
    verifyEqual(testCase, neuramed.evaluation.computeAccuracy(Y, Y), 1.0);
end

function test_computeAccuracy_allWrong(testCase)
    Ytrue = categorical({'A';'B'});
    Ypred = categorical({'B';'A'});
    verifyEqual(testCase, neuramed.evaluation.computeAccuracy(Ytrue, Ypred), 0.0);
end

function test_confusionMatrix_diagonalMatchesCorrectPredictions(testCase)
    Ytrue = categorical({'A';'A';'B';'B'});
    Ypred = categorical({'A';'B';'B';'B'});
    [C, names] = neuramed.evaluation.computeConfusionMatrix(Ytrue, Ypred);
    verifyEqual(testCase, sum(diag(C)), 3);
    verifyEqual(testCase, sort(names), sort({'A';'B'}));
end

function test_precisionRecall_knownValues(testCase)
    % TP=3 FP=1 FN=1 for class 'Pos'
    Ytrue = categorical([repmat({'Pos'},4,1); repmat({'Neg'},4,1)]);
    Ypred = categorical({'Pos';'Pos';'Pos';'Neg';'Pos';'Neg';'Neg';'Neg'});
    metrics = neuramed.evaluation.computePrecisionRecall(Ytrue, Ypred);
    posRow = metrics(strcmp(string(metrics.Class), 'Pos'), :);
    verifyEqual(testCase, posRow.Precision, 3/4, 'AbsTol', 1e-10);
    verifyEqual(testCase, posRow.Recall, 3/4, 'AbsTol', 1e-10);
end

function test_f1Score_harmonicMeanCorrect(testCase)
    Ytrue = categorical([repmat({'Pos'},4,1); repmat({'Neg'},4,1)]);
    Ypred = categorical({'Pos';'Pos';'Pos';'Neg';'Pos';'Neg';'Neg';'Neg'});
    metrics = neuramed.evaluation.computeF1Score(Ytrue, Ypred);
    posRow = metrics(strcmp(string(metrics.Class), 'Pos'), :);
    expectedF1 = 2 * (0.75 * 0.75) / (0.75 + 0.75);
    verifyEqual(testCase, posRow.F1Score, expectedF1, 'AbsTol', 1e-10);
end

function test_sensitivitySpecificity_sumToSensibleRange(testCase)
    Ytrue = categorical([repmat({'Pos'},10,1); repmat({'Neg'},10,1)]);
    Ypred = Ytrue; Ypred(1:2) = categorical({'Neg';'Neg'}); % 2 false negatives
    metrics = neuramed.evaluation.computeSpecificitySensitivity(Ytrue, Ypred);
    posRow = metrics(strcmp(string(metrics.Class), 'Pos'), :);
    verifyEqual(testCase, posRow.Sensitivity, 0.8, 'AbsTol', 1e-10);
end

function test_computeROCAUC_perfectSeparationGivesAUC1(testCase)
    Ytrue = categorical([zeros(50,1); ones(50,1)], [0 1], {'Neg','Pos'});
    scores = [zeros(50,1); ones(50,1)]; % perfectly separated scores
    result = neuramed.evaluation.computeROCAUC(Ytrue, scores);
    verifyEqual(testCase, result.AUC, 1.0, 'AbsTol', 1e-10);
end

function test_computeROCAUC_randomScoresNearHalf(testCase)
    rng(11);
    Ytrue = categorical(randi([0 1], 500, 1), [0 1], {'Neg','Pos'});
    scores = rand(500, 1); % uninformative
    result = neuramed.evaluation.computeROCAUC(Ytrue, scores);
    verifyEqual(testCase, result.AUC, 0.5, 'AbsTol', 0.1);
end

function test_computeROCAUC_nonBinaryThrows(testCase)
    Ytrue = categorical({'A';'B';'C'});
    scores = [0.1; 0.5; 0.9];
    verifyError(testCase, @() neuramed.evaluation.computeROCAUC(Ytrue, scores), ...
        'neuramed:evaluation:computeROCAUC:NotBinary');
end

function test_generateEvaluationReport_hasAllFields(testCase)
    Ytrue = categorical([zeros(20,1); ones(20,1)], [0 1], {'Neg','Pos'});
    Ypred = Ytrue;
    probs = [1-double(Ytrue=='Pos'), double(Ytrue=='Pos')];
    report = neuramed.evaluation.generateEvaluationReport(Ytrue, Ypred, probs);
    verifyTrue(testCase, isfield(report, 'Accuracy'));
    verifyTrue(testCase, isfield(report, 'AUC'));
    verifyTrue(testCase, isfield(report, 'MacroF1'));
    verifyEqual(testCase, report.Accuracy, 1.0);
end

function test_crossValidateModel_returnsAggregatedStats(testCase)
    rng(9);
    N = 100;
    X = [randn(N/2,3)+2; randn(N/2,3)-2];
    Y = categorical([ones(N/2,1); 2*ones(N/2,1)], [1 2], {'A','B'});
    T = array2table(X);
    data = struct('X', T, 'Y', Y, 'Type', 'tabular', 'Source', 'synthetic', 'Metadata', struct());
    cvResults = neuramed.evaluation.crossValidateModel(@neuramed.models.trainDecisionTree, data, 'K', 4);
    verifyEqual(testCase, cvResults.K, 4);
    verifyGreaterThanOrEqual(testCase, cvResults.MeanAccuracy, 0.7);
end
