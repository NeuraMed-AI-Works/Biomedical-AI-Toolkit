function tests = test_utils_module
%TEST_UTILS_MODULE Unit tests for neuramed.utils.*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root, 'startup.m'));
end

function test_manageVersion_getReturnsValidSemver(testCase)
    v = neuramed.utils.manageVersion('get');
    verifyTrue(testCase, ~isempty(regexp(v, '^\d+\.\d+\.\d+', 'once')));
end

function test_saveAndLoadModel_roundTrip(testCase)
    rng(30);
    Xtr = [randn(30,3)+1; randn(30,3)-1];
    Ytr = categorical([ones(30,1); 2*ones(30,1)], [1 2], {'A','B'});
    model = neuramed.models.trainDecisionTree(Xtr, Ytr);

    tmpFile = fullfile(tempdir, 'neuramed_test_model.mat');
    if exist(tmpFile, 'file'), delete(tmpFile); end

    neuramed.utils.saveModel(model, tmpFile);
    loaded = neuramed.utils.loadModel(tmpFile);

    verifyEqual(testCase, loaded.Type, model.Type);

    pred1 = neuramed.models.predictModel(model, Xtr);
    pred2 = neuramed.models.predictModel(loaded, Xtr);
    verifyEqual(testCase, pred1, pred2);

    delete(tmpFile);
end

function test_saveModel_refusesOverwriteByDefault(testCase)
    rng(31);
    Xtr = randn(10,2); Ytr = categorical(randi([1 2],10,1), [1 2], {'A','B'});
    model = neuramed.models.trainDecisionTree(Xtr, Ytr);
    tmpFile = fullfile(tempdir, 'neuramed_test_model2.mat');
    neuramed.utils.saveModel(model, tmpFile, 'Overwrite', true);
    verifyError(testCase, @() neuramed.utils.saveModel(model, tmpFile), ...
        'neuramed:utils:saveModel:FileExists');
    delete(tmpFile);
end

function test_batchPredict_matchesDirectPrediction(testCase)
    rng(32);
    Xtr = [randn(40,3)+2; randn(40,3)-2];
    Ytr = categorical([ones(40,1); 2*ones(40,1)], [1 2], {'A','B'});
    model = neuramed.models.trainRandomForest(Xtr, Ytr, 'NumTrees', 20);

    Xnew = randn(53, 3); % not evenly divisible by batch size
    directPred = neuramed.models.predictModel(model, Xnew);
    results = neuramed.utils.batchPredict(model, Xnew, 'BatchSize', 10, 'ShowProgress', false);

    verifyEqual(testCase, results.Predictions, directPred);
    verifyEqual(testCase, results.NumProcessed, 53);
end

function test_manageConfig_saveLoadRoundTrip(testCase)
    cfg = struct('ModelType', 'RandomForest', 'NumTrees', 100, 'Threshold', 0.5);
    tmpFile = fullfile(tempdir, 'neuramed_test_config.json');
    neuramed.utils.manageConfig('save', tmpFile, cfg);
    loaded = neuramed.utils.manageConfig('load', tmpFile);
    verifyEqual(testCase, loaded.ModelType, cfg.ModelType);
    verifyEqual(testCase, loaded.NumTrees, cfg.NumTrees);
    delete(tmpFile);
end

function test_manageConfig_getMissingKeyThrows(testCase)
    cfg = struct('A', 1);
    verifyError(testCase, @() neuramed.utils.manageConfig('get', cfg, 'B'), ...
        'neuramed:utils:manageConfig:MissingKey');
end

function test_logPerformance_appendsToFile(testCase)
    tmpLog = fullfile(tempdir, 'neuramed_test_perf.jsonl');
    if exist(tmpLog, 'file'), delete(tmpLog); end
    neuramed.utils.logPerformance('UnitTestEvent', struct('Accuracy', 0.9), 'LogFile', tmpLog);
    verifyTrue(testCase, exist(tmpLog, 'file') == 2);
    content = fileread(tmpLog);
    verifyTrue(testCase, contains(content, 'UnitTestEvent'));
    delete(tmpLog);
end
