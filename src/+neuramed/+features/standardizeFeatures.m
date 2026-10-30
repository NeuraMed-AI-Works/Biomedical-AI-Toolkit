function [Xs, params] = standardizeFeatures(X, options)
%STANDARDIZEFEATURES Z-score standardize features (zero mean, unit variance).
%
%   [XS, PARAMS] = neuramed.features.standardizeFeatures(X) fits mean/std
%   on X and returns standardized output plus fitted parameters.
%
%   XS = neuramed.features.standardizeFeatures(X, 'Params', PARAMS) applies
%   previously fitted PARAMS to new data (fit-on-train-only discipline;
%   never refit on test/validation data).
%
%   Mathematical Definition:
%       x' = (x - mu) / sigma
%       mu    = (1/N) * sum(x_i)
%       sigma = sqrt( (1/(N-1)) * sum((x_i - mu)^2) )   (sample std, ddof=1)
%
%       Columns with sigma < EPSILON are left at (x - mu) to avoid
%       division blow-up (they carry no discriminative variance anyway).
%
%   Example:
%       [Xtr_s, p] = neuramed.features.standardizeFeatures(Xtrain);
%       Xte_s = neuramed.features.standardizeFeatures(Xtest, 'Params', p);
%
%   See also NEURAMED.FEATURES.NORMALIZEFEATURES

    arguments
        X
        options.Params = []
    end

    EPSILON = 1e-12;
    [Xmat, isTable, varNames] = neuramed.features.internal.toMatrix(X);

    if isempty(options.Params)
        mu = mean(Xmat, 1, 'omitnan');
        sigma = std(Xmat, 0, 1, 'omitnan');
        params = struct('Mean', mu, 'Std', sigma);
    else
        params = options.Params;
        mu = params.Mean;
        sigma = params.Std;
    end

    safeSigma = max(sigma, EPSILON);
    Xstd = (Xmat - mu) ./ safeSigma;

    Xs = neuramed.features.internal.fromMatrix(Xstd, isTable, varNames, X);
end
