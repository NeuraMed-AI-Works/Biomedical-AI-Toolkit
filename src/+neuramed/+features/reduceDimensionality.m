function [Xreduced, params] = reduceDimensionality(X, options)
%REDUCEDIMENSIONALITY Unified interface to dimensionality reduction methods.
%
%   [XR, PARAMS] = neuramed.features.reduceDimensionality(X) reduces the
%   dimensionality of X using the selected 'Method'. Serves as a single
%   dispatch point so pipelines can swap methods via configuration
%   without changing calling code.
%
%   Name-Value Options:
%       'Method'  - 'pca' | 'tsne' | 'umap-like' (default 'pca')
%       'NumComponents' - target dimensionality (default 2 for 'tsne', else 'auto')
%       'Y'       - labels, required only if a future supervised method is used
%       'Params'  - previously fitted params (pca only; tsne is transductive)
%
%   Mathematical Notes:
%       'pca' — see neuramed.features.applyPCA (linear, variance-preserving).
%       'tsne' — t-Distributed Stochastic Neighbor Embedding: models
%       pairwise similarities in high-D as Gaussian conditional
%       probabilities p_{j|i} and in low-D as Student-t distributed
%       probabilities q_{j|i}, then minimizes the Kullback-Leibler
%       divergence KL(P||Q) = sum_i sum_j p_ij * log(p_ij/q_ij) via
%       gradient descent. Excellent for visualization; not a stable
%       fit/transform (no out-of-sample projection), so it is only
%       applied to the full dataset provided.
%
%   Example:
%       Xr = neuramed.features.reduceDimensionality(X, 'Method', 'pca', 'NumComponents', 10);
%       Xviz = neuramed.features.reduceDimensionality(X, 'Method', 'tsne');
%
%   See also NEURAMED.FEATURES.APPLYPCA

    arguments
        X
        options.Method (1,:) char {mustBeMember(options.Method, {'pca','tsne'})} = 'pca'
        options.NumComponents = []
        options.Y = []
        options.Params = []
    end

    switch options.Method
        case 'pca'
            nc = options.NumComponents;
            if isempty(nc)
                [Xreduced, params] = neuramed.features.applyPCA(X, 'Params', options.Params);
            else
                [Xreduced, params] = neuramed.features.applyPCA(X, 'NumComponents', nc, 'Params', options.Params);
            end
        case 'tsne'
            [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
            nc = options.NumComponents;
            if isempty(nc), nc = 2; end
            Xreduced = tsne(Xmat, 'NumDimensions', nc);
            params = struct('Method', 'tsne', 'Note', 'Transductive: refit required for new data.');
    end
end
