function [Xpca, params] = applyPCA(X, options)
%APPLYPCA Principal Component Analysis for linear dimensionality reduction.
%
%   [XPCA, PARAMS] = neuramed.features.applyPCA(X) fits PCA on X and
%   projects it onto principal components explaining a target variance.
%
%   XPCA = neuramed.features.applyPCA(X, 'Params', PARAMS) projects new
%   data using previously fitted PARAMS (fit-on-train-only).
%
%   Name-Value Options:
%       'NumComponents'    - fixed number of components (overrides VarianceExplained)
%       'VarianceExplained'- target cumulative variance fraction (default 0.95)
%       'Params'           - previously fitted PCA params
%
%   Mathematical Definition:
%       Given centered data Xc = X - mean(X), PCA computes the
%       eigendecomposition of the covariance matrix:
%           C = (1/(N-1)) * Xc' * Xc = V * D * V'
%       where columns of V are eigenvectors (principal directions) and D
%       is diagonal with eigenvalues lambda_1 >= lambda_2 >= ... .
%       The projection onto the first k components is:
%           Xpca = Xc * V(:, 1:k)
%       Cumulative variance explained by k components:
%           r(k) = sum_{i=1}^{k} lambda_i / sum_{i=1}^{D} lambda_i
%       k is chosen as the smallest value with r(k) >= VarianceExplained.
%
%   Example:
%       [Xtr_p, p] = neuramed.features.applyPCA(Xtrain, 'VarianceExplained', 0.95);
%       Xte_p = neuramed.features.applyPCA(Xtest, 'Params', p);
%
%   See also NEURAMED.FEATURES.APPLYLDA, NEURAMED.FEATURES.REDUCEDIMENSIONALITY

    arguments
        X
        options.NumComponents = []
        options.VarianceExplained (1,1) double {mustBeInRange(options.VarianceExplained, 0, 1)} = 0.95
        options.Params = []
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);

    if isempty(options.Params)
        mu = mean(Xmat, 1, 'omitnan');
        Xc = Xmat - mu;
        [coeff, ~, latent, ~, explained] = pca(Xc);
        if isempty(options.NumComponents)
            cumExplained = cumsum(explained) / 100;
            k = find(cumExplained >= options.VarianceExplained, 1, 'first');
            if isempty(k), k = numel(explained); end
        else
            k = min(options.NumComponents, size(coeff, 2));
        end
        params = struct('Mean', mu, 'Coeff', coeff(:, 1:k), 'Latent', latent, ...
            'Explained', explained, 'NumComponents', k);
    else
        params = options.Params;
        Xc = Xmat - params.Mean;
    end

    Xpca = Xc * params.Coeff;
end
