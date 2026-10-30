function result = predictDisease(model, X, options)
%PREDICTDISEASE Run disease prediction with clinical-facing output.
%
%   RESULT = neuramed.clinical.predictDisease(MODEL, X) runs MODEL on X
%   and packages the outcome into a clinically interpretable struct
%   (predicted class, confidence, probability distribution) rather than
%   a bare label — the entry point clinical UIs should call.
%
%   Name-Value Options:
%       'ClassLabelsMap' - containers.Map from raw class name to a
%                           human-readable clinical label
%
%   Output RESULT fields (one entry per observation, struct array):
%       .PredictedClass, .DisplayLabel, .Confidence, .ProbabilityDistribution
%
%   Example:
%       result = neuramed.clinical.predictDisease(model, patientFeatures);
%       fprintf('%s (confidence %.1f%%)\n', result(1).DisplayLabel, result(1).Confidence*100);
%
%   See also NEURAMED.CLINICAL.COMPUTECONFIDENCESCORE, NEURAMED.CLINICAL.CLASSIFYRISK

    arguments
        model
        X
        options.ClassLabelsMap = []
    end

    [preds, probs] = neuramed.models.predictModel(model, X);
    confidence = neuramed.clinical.computeConfidenceScore(probs);

    n = numel(preds);
    result = repmat(struct('PredictedClass', [], 'DisplayLabel', '', ...
        'Confidence', 0, 'ProbabilityDistribution', []), n, 1);

    for i = 1:n
        rawClass = preds(i);
        result(i).PredictedClass = rawClass;
        if ~isempty(options.ClassLabelsMap) && isKey(options.ClassLabelsMap, char(rawClass))
            result(i).DisplayLabel = options.ClassLabelsMap(char(rawClass));
        else
            result(i).DisplayLabel = char(rawClass);
        end
        result(i).Confidence = confidence(i);
        result(i).ProbabilityDistribution = probs(i, :);
    end
end
