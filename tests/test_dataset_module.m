function tests = test_dataset_module
%TEST_DATASET_MODULE Unit tests for neuramed.dataset.*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root, 'startup.m'));
end

function data = makeSyntheticData()
    rng(0);
    N = 100;
    X = array2table([randn(N,1), randn(N,1)*2 + 1], 'VariableNames', {'F1','F2'});
    Y = categorical(rand(N,1) > 0.5, [false true], {'Neg','Pos'});
    data = struct('X', X, 'Y', Y, 'Type', 'tabular', 'Source', 'synthetic', 'Metadata', struct());
end

function test_splitDataset_fractionsSumToTotal(testCase)
    data = makeSyntheticData();
    [trainD, testD] = neuramed.dataset.splitDataset(data, 'TrainFraction', 0.8);
    total = height(trainD.X) + height(testD.X);
    verifyEqual(testCase, total, height(data.X));
end

function test_splitDataset_stratificationApprox(testCase)
    data = makeSyntheticData();
    [trainD, ~] = neuramed.dataset.splitDataset(data, 'TrainFraction', 0.8, 'Stratify', true);
    overallRatio = mean(data.Y == 'Pos');
    trainRatio = mean(trainD.Y == 'Pos');
    verifyEqual(testCase, trainRatio, overallRatio, 'AbsTol', 0.15);
end

function test_splitDataset_invalidFractionsThrows(testCase)
    data = makeSyntheticData();
    verifyError(testCase, @() neuramed.dataset.splitDataset(data, ...
        'TrainFraction', 0.8, 'ValFraction', 0.5), ...
        'neuramed:dataset:splitDataset:FractionsInvalid');
end

function test_crossValidationSplit_correctFoldCount(testCase)
    data = makeSyntheticData();
    folds = neuramed.dataset.crossValidationSplit(data, 'K', 5);
    verifyEqual(testCase, numel(folds), 5);
end

function test_crossValidationSplit_noOverlapBetweenFolds(testCase)
    data = makeSyntheticData();
    folds = neuramed.dataset.crossValidationSplit(data, 'K', 4);
    totalTest = 0;
    for k = 1:numel(folds)
        totalTest = totalTest + height(folds(k).Test.X);
    end
    verifyEqual(testCase, totalTest, height(data.X));
end

function test_validateData_detectsMissingValues(testCase)
    data = makeSyntheticData();
    data.X.F1(1:10) = NaN;
    report = neuramed.dataset.validateData(data, 'MissingThreshold', 0.05);
    verifyFalse(testCase, report.IsValid);
    verifyGreaterThan(testCase, report.MissingSummary.MissingCount(1), 0);
end

function test_validateData_strictModeThrows(testCase)
    data = makeSyntheticData();
    data.X.F1(1:50) = NaN;
    verifyError(testCase, @() neuramed.dataset.validateData(data, 'Strict', true), ...
        'neuramed:dataset:validateData:ValidationFailed');
end

function test_augmentData_multipliesRowCount(testCase)
    data = makeSyntheticData();
    aug = neuramed.dataset.augmentData(data, 'Multiplier', 3);
    verifyEqual(testCase, height(aug.X), height(data.X) * 3);
end

function test_manageMetadata_createAndVerify(testCase)
    data = makeSyntheticData();
    meta = neuramed.dataset.manageMetadata('create', data);
    isUnchanged = neuramed.dataset.manageMetadata('verify', data, meta);
    verifyTrue(testCase, isUnchanged);
end
