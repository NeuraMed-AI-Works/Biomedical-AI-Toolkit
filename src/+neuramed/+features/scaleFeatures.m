function [Xs, params] = scaleFeatures(X, options)
%SCALEFEATURES Robust scaling using median and interquartile range (IQR).
%
%   [XS, PARAMS] = neuramed.features.scaleFeatures(X) scales features
%   using the median/IQR estimator, which is robust to outliers commonly
%   present in clinical measurements (e.g. lab values with extreme
%   pathological readings).
%
%   XS = neuramed.features.scaleFeatures(X, 'Params', PARAMS) applies
%   previously fitted PARAMS.
%
%   Mathematical Definition:
%       x' = (x - median(x)) / IQR(x),   IQR(x) = Q3(x) - Q1(x)
%
%   Example:
%       [Xtr_s, p] = neuramed.features.scaleFeatures(Xtrain);
%       Xte_s = neuramed.features.scaleFeatures(Xtest, 'Params', p);
%
%   See also NEURAMED.FEATURES.STANDARDIZEFEATURES, NEURAMED.FEATURES.NORMALIZEFEATURES

    arguments
        X
        options.Params = []
    end

    EPSILON = 1e-12;
    [Xmat, isTable, varNames] = neuramed.features.internal.toMatrix(X);

    if isempty(options.Params)
        med = median(Xmat, 1, 'omitnan');
        q1 = prctile(Xmat, 25, 1);
        q3 = prctile(Xmat, 75, 1);
        iqr_ = q3 - q1;
        params = struct('Median', med, 'IQR', iqr_);
    else
        params = options.Params;
        med = params.Median;
        iqr_ = params.IQR;
    end

    safeIQR = max(iqr_, EPSILON);
    Xscaled = (Xmat - med) ./ safeIQR;

    Xs = neuramed.features.internal.fromMatrix(Xscaled, isTable, varNames, X);
end
