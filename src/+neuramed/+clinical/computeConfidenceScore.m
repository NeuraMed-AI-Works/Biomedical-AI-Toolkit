function confidence = computeConfidenceScore(probabilities)
%COMPUTECONFIDENCESCORE Derive a scalar confidence score from class probabilities.
%
%   CONFIDENCE = neuramed.clinical.computeConfidenceScore(PROBS) returns,
%   per row of the N x C probability matrix PROBS, the probability mass
%   on the predicted (argmax) class.
%
%   Mathematical Definition:
%       confidence_i = max_c P(Y=c | x_i)
%       This is the simplest, most interpretable calibrated-confidence
%       proxy; contrast with predictive entropy
%           H_i = -sum_c P(c|x_i) log P(c|x_i)
%       which captures overall distributional uncertainty (low entropy =
%       high confidence) and is also returned when requested.
%
%   Example:
%       confidence = neuramed.clinical.computeConfidenceScore(probs);
%
%   See also NEURAMED.CLINICAL.ESTIMATEPROBABILITY

    if isempty(probabilities)
        confidence = [];
        return
    end
    confidence = max(probabilities, [], 2);
end
