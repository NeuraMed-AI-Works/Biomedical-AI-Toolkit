function tests = test_features_module
%TEST_FEATURES_MODULE Unit tests for neuramed.features.*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root, 'startup.m'));
end

function test_normalizeFeatures_rangeIsZeroOne(testCase)
    X = [1 10; 5 20; 10 30];
    Xn = neuramed.features.normalizeFeatures(X);
    verifyEqual(testCase, min(Xn(:)), 0, 'AbsTol', 1e-10);
    verifyEqual(testCase, max(Xn(:)), 1, 'AbsTol', 1e-10);
end

function test_normalizeFeatures_paramsReusable(testCase)
    Xtrain = [1 10; 5 20; 10 30];
    Xtest = [3 15; 8 25];
    [~, params] = neuramed.features.normalizeFeatures(Xtrain);
    Xtest_n = neuramed.features.normalizeFeatures(Xtest, 'Params', params);
    verifySize(testCase, Xtest_n, size(Xtest));
end

function test_standardizeFeatures_zeroMeanUnitVar(testCase)
    rng(1);
    X = randn(1000, 3) * 5 + 10;
    Xs = neuramed.features.standardizeFeatures(X);
    verifyEqual(testCase, mean(Xs, 1), zeros(1,3), 'AbsTol', 1e-8);
    verifyEqual(testCase, std(Xs, 0, 1), ones(1,3), 'AbsTol', 1e-8);
end

function test_scaleFeatures_robustToOutliers(testCase)
    X = [1;2;3;4;5;1000];
    Xs = neuramed.features.scaleFeatures(X);
    % median-based scaling should keep the bulk of values in a modest range
    verifyLessThan(testCase, abs(Xs(1)), 5);
end

function test_applyPCA_reducesDimensionality(testCase)
    rng(2);
    X = randn(200, 10);
    X(:, 5:10) = X(:, 1:2) * randn(2,6) + 0.01*randn(200,6); % redundant features
    [Xp, params] = neuramed.features.applyPCA(X, 'VarianceExplained', 0.9);
    verifyLessThan(testCase, params.NumComponents, 10);
    verifyEqual(testCase, size(Xp, 2), params.NumComponents);
end

function test_applyPCA_paramsReusableOnNewData(testCase)
    rng(3);
    Xtrain = randn(100, 5);
    Xtest = randn(20, 5);
    [~, params] = neuramed.features.applyPCA(Xtrain, 'NumComponents', 3);
    Xtest_p = neuramed.features.applyPCA(Xtest, 'Params', params);
    verifyEqual(testCase, size(Xtest_p), [20 3]);
end

function test_applyLDA_outputDimMatchesClassesMinusOne(testCase)
    rng(4);
    X = [randn(50,4); randn(50,4)+3; randn(50,4)-3];
    Y = categorical([ones(50,1); 2*ones(50,1); 3*ones(50,1)]);
    [Xl, params] = neuramed.features.applyLDA(X, Y);
    verifyEqual(testCase, size(Xl, 2), 2); % min(C-1, D) = min(2,4) = 2
end

function test_rankFeatures_returnsSortedDescending(testCase)
    rng(5);
    X = randn(200, 6);
    Y = categorical(X(:,1) + randn(200,1)*0.1 > 0);
    ranking = neuramed.features.rankFeatures(X, Y, 'Method', 'fscore');
    verifyTrue(testCase, issorted(ranking.Score, 'descend'));
end

function test_selectFeatures_returnsRequestedCount(testCase)
    rng(6);
    X = randn(150, 8);
    Y = categorical(X(:,1) + randn(150,1)*0.2 > 0);
    [Xsel, names] = neuramed.features.selectFeatures(X, Y, 'NumFeatures', 3, 'Method', 'fscore');
    verifyEqual(testCase, size(Xsel, 2), 3);
    verifyEqual(testCase, numel(names), 3);
end
