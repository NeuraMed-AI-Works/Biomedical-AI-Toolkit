function [predictions, probabilities, scores] = predictModel(model, X)
%PREDICTMODEL Unified prediction interface for all NeuraMed model types.
%
%   PRED = neuramed.models.predictModel(MODEL, X) returns predicted class
%   labels for X using any model produced by the neuramed.models.train*
%   functions (single dispatch point, so downstream code never needs to
%   know which underlying algorithm was used).
%
%   [PRED, PROBS] = neuramed.models.predictModel(MODEL, X) also returns
%   an N x C matrix of class posterior probabilities.
%
%   [PRED, PROBS, SCORES] = ... additionally returns raw classification
%   scores (model-dependent; not always probabilistic).
%
%   Example:
%       pred = neuramed.models.predictModel(model, Xtest);
%       [pred, probs] = neuramed.models.predictModel(model, Xtest);
%
%   See also NEURAMED.MODELS.TRAINRANDOMFOREST, NEURAMED.CLINICAL.PREDICTDISEASE

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);

    switch model.Type
        case 'Ensemble'
            [predictions, probabilities, scores] = neuramed.models.internal.predictEnsemble(model, Xmat);
        otherwise
            [predictions, scores] = predict(model.Underlying, Xmat);
            probabilities = scores;
    end
end
