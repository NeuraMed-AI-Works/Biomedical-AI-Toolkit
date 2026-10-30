function tests = test_models_module
%TEST_MODELS_MODULE Unit tests for neuramed.models.*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root, 'startup.m'));
end

function [Xtrain, Ytrain, Xtest, Ytest] = makeSeparableData()
    rng(42);
    N = 200;
    X1 = randn(N/2, 4) + 2;
    X2 = randn(N/2, 4) - 2;
    X = [X1; X2];
    Y = categorical([ones(N/2,1); 2*ones(N/2,1)], [1 2], {'ClassA','ClassB'});
    idx = randperm(N);
    X = X(idx, :); Y = Y(idx);
    Xtrain = X(1:150, :); Ytrain = Y(1:150);
    Xtest = X(151:end, :); Ytest = Y(151:end);
end

function test_decisionTree_achievesGoodAccuracyOnSeparableData(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainDecisionTree(Xtr, Ytr);
    pred = neuramed.models.predictModel(model, Xte);
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_randomForest_achievesGoodAccuracy(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainRandomForest(Xtr, Ytr, 'NumTrees', 50);
    pred = neuramed.models.predictModel(model, Xte);
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_svm_achievesGoodAccuracy(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainSVM(Xtr, Ytr, 'KernelFunction', 'gaussian');
    pred = neuramed.models.predictModel(model, Xte);
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_knn_achievesGoodAccuracy(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainKNN(Xtr, Ytr, 'NumNeighbors', 5);
    pred = neuramed.models.predictModel(model, Xte);
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_naiveBayes_achievesGoodAccuracy(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainNaiveBayes(Xtr, Ytr);
    pred = neuramed.models.predictModel(model, Xte);
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_logisticRegression_achievesGoodAccuracy(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainLogisticRegression(Xtr, Ytr);
    pred = neuramed.models.predictModel(model, Xte);
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_xgboost_achievesGoodAccuracy(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainXGBoost(Xtr, Ytr, 'NumTrees', 50);
    pred = neuramed.models.predictModel(model, Xte);
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_ensemble_softVoteProducesValidPredictions(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainEnsemble(Xtr, Ytr, ...
        'BaseLearners', {'tree','knn','logistic'}, 'CombineMethod', 'softvote');
    pred = neuramed.models.predictModel(model, Xte);
    verifyEqual(testCase, numel(pred), numel(Yte));
    acc = neuramed.evaluation.computeAccuracy(Yte, pred);
    verifyGreaterThanOrEqual(testCase, acc, 0.85);
end

function test_ensemble_stackingProducesValidPredictions(testCase)
    [Xtr, Ytr, Xte, Yte] = makeSeparableData();
    model = neuramed.models.trainEnsemble(Xtr, Ytr, ...
        'BaseLearners', {'tree','logistic'}, 'CombineMethod', 'stacking');
    pred = neuramed.models.predictModel(model, Xte);
    verifyEqual(testCase, numel(pred), numel(Yte));
end

function test_trainEnsemble_invalidLearnerThrows(testCase)
    [Xtr, Ytr, ~, ~] = makeSeparableData();
    verifyError(testCase, @() neuramed.models.trainEnsemble(Xtr, Ytr, ...
        'BaseLearners', {'not_a_real_learner'}), ...
        'neuramed:models:trainEnsemble:UnknownLearner');
end
