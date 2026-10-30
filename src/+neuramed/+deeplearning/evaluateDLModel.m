function results = evaluateDLModel(net, testData, trueLabels)
%EVALUATEDLMODEL Evaluate a trained deep network on held-out image data.
%
%   RESULTS = neuramed.deeplearning.evaluateDLModel(NET, TESTDATA, TRUELABELS)
%   runs inference and computes the standard NeuraMed evaluation report
%   (see neuramed.evaluation.generateEvaluationReport), reusing the same
%   metric definitions as the classical ML models module for consistency
%   across the toolkit.
%
%   Example:
%       results = neuramed.deeplearning.evaluateDLModel(net, augTest, testImds.Labels);
%
%   See also NEURAMED.EVALUATION.GENERATEEVALUATIONREPORT

    [predictedLabels, scores] = classify(net, testData);
    results = neuramed.evaluation.generateEvaluationReport(trueLabels, predictedLabels, scores);
end
