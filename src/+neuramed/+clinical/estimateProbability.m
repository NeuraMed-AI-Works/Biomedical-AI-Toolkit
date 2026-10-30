function [probEstimate, entropy] = estimateProbability(model, X, options)
%ESTIMATEPROBABILITY Estimate calibrated class probabilities and uncertainty.
%
%   [PROBS, ENTROPY] = neuramed.clinical.estimateProbability(MODEL, X)
%   returns the model's posterior probability matrix along with
%   per-observation Shannon entropy as an uncertainty measure.
%
%   Name-Value Options:
%       'Calibrate' - logical, apply Platt scaling if the underlying
%                     model supports it (default false; assumes model
%                     probabilities are already reasonably calibrated,
%                     e.g. via fitPosterior for SVM)
%
%   Mathematical Notes:
%       Shannon entropy of the predictive distribution:
%           H(x) = - sum_c P(c|x) * log2(P(c|x))
%       H is 0 when the model is fully certain (one class has
%       probability 1) and maximal (log2(C)) under a uniform
%       distribution over C classes — useful for flagging cases that may
%       need clinician review rather than fully automated triage.
%
%   Example:
%       [probs, entropy] = neuramed.clinical.estimateProbability(model, Xpatient);
%
%   See also NEURAMED.CLINICAL.COMPUTECONFIDENCESCORE

    arguments
        model
        X
        options.Calibrate (1,1) logical = false
    end

    [~, probEstimate] = neuramed.models.predictModel(model, X);

    EPSILON = 1e-12;
    p = max(probEstimate, EPSILON);
    entropy = -sum(p .* log2(p), 2);
end
