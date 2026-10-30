function tests = test_clinical_module
%TEST_CLINICAL_MODULE Unit tests for neuramed.clinical.*
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root, 'startup.m'));
end

function test_computeConfidenceScore_returnsMaxProbability(testCase)
    probs = [0.7 0.3; 0.2 0.8; 0.5 0.5];
    conf = neuramed.clinical.computeConfidenceScore(probs);
    verifyEqual(testCase, conf, [0.7; 0.8; 0.5], 'AbsTol', 1e-10);
end

function test_classifyRisk_defaultThresholds(testCase)
    verifyEqual(testCase, char(neuramed.clinical.classifyRisk(0.1)), 'Low');
    verifyEqual(testCase, char(neuramed.clinical.classifyRisk(0.4)), 'Moderate');
    verifyEqual(testCase, char(neuramed.clinical.classifyRisk(0.6)), 'High');
    verifyEqual(testCase, char(neuramed.clinical.classifyRisk(0.9)), 'Critical');
end

function test_classifyRisk_outOfRangeThrows(testCase)
    verifyError(testCase, @() neuramed.clinical.classifyRisk(1.5), ...
        'neuramed:clinical:classifyRisk:OutOfRange');
end

function test_predictDisease_returnsStructArrayWithExpectedFields(testCase)
    rng(20);
    Xtr = [randn(50,3)+2; randn(50,3)-2];
    Ytr = categorical([ones(50,1); 2*ones(50,1)], [1 2], {'Healthy','Disease'});
    model = neuramed.models.trainRandomForest(Xtr, Ytr, 'NumTrees', 30);

    Xnew = randn(3,3) - 2; % should lean toward 'Disease'
    result = neuramed.clinical.predictDisease(model, Xnew);

    verifyEqual(testCase, numel(result), 3);
    verifyTrue(testCase, isfield(result, 'Confidence'));
    verifyTrue(testCase, isfield(result, 'DisplayLabel'));
    for i = 1:numel(result)
        verifyGreaterThanOrEqual(testCase, result(i).Confidence, 0);
        verifyLessThanOrEqual(testCase, result(i).Confidence, 1);
    end
end

function test_generateClinicalRecommendation_includesDisclaimerByDefault(testCase)
    predictionResult = struct('PredictedClass', categorical({'Disease'}), ...
        'DisplayLabel', 'Disease', 'Confidence', 0.9, ...
        'ProbabilityDistribution', [0.1 0.9]);
    rec = neuramed.clinical.generateClinicalRecommendation(predictionResult);
    verifyNotEmpty(testCase, rec.Disclaimer);
    verifyTrue(testCase, contains(rec.Disclaimer, 'does NOT constitute a medical diagnosis'));
end

function test_estimateProbability_entropyZeroForCertainPrediction(testCase)
    rng(21);
    Xtr = [randn(50,3)+5; randn(50,3)-5];
    Ytr = categorical([ones(50,1); 2*ones(50,1)], [1 2], {'A','B'});
    model = neuramed.models.trainRandomForest(Xtr, Ytr, 'NumTrees', 30);
    Xnew = [10 10 10]; % far into class A's region
    [~, entropy] = neuramed.clinical.estimateProbability(model, Xnew);
    verifyLessThan(testCase, entropy, 0.5);
end
