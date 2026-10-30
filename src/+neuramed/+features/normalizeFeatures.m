function [Xn, params] = normalizeFeatures(X, options)
%NORMALIZEFEATURES Min-max normalize features to a target range.
%
%   [XN, PARAMS] = neuramed.features.normalizeFeatures(X) scales each
%   column of numeric matrix/table X to [0, 1].
%
%   XN = neuramed.features.normalizeFeatures(X, 'Params', PARAMS) applies
%   previously fitted PARAMS (e.g. from training data) to new data X,
%   which is essential to avoid test-set leakage.
%
%   Name-Value Options:
%       'Range'  - target range [lo hi] (default [0 1])
%       'Params' - previously fitted params struct (skips fitting)
%
%   Mathematical Definition:
%       x' = lo + (x - min(x)) / (max(x) - min(x)) * (hi - lo)
%
%       Columns with max(x) == min(x) are mapped to lo to avoid
%       division by zero (EPSILON-guarded).
%
%   Example:
%       [Xtrain_n, p] = neuramed.features.normalizeFeatures(Xtrain);
%       Xtest_n = neuramed.features.normalizeFeatures(Xtest, 'Params', p);
%
%   See also NEURAMED.FEATURES.STANDARDIZEFEATURES

    arguments
        X
        options.Range (1,2) double = [0 1]
        options.Params = []
    end

    EPSILON = 1e-12;
    [Xmat, isTable, varNames] = neuramed.features.internal.toMatrix(X);

    if isempty(options.Params)
        colMin = min(Xmat, [], 1, 'omitnan');
        colMax = max(Xmat, [], 1, 'omitnan');
        params = struct('Min', colMin, 'Max', colMax, 'Range', options.Range);
    else
        params = options.Params;
        colMin = params.Min;
        colMax = params.Max;
    end

    denom = max(colMax - colMin, EPSILON);
    lo = params.Range(1); hi = params.Range(2);
    Xscaled = lo + (Xmat - colMin) ./ denom * (hi - lo);

    Xn = neuramed.features.internal.fromMatrix(Xscaled, isTable, varNames, X);
end
