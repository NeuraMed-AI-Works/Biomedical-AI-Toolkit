function [Xlda, params] = applyLDA(X, Y, options)
%APPLYLDA Linear Discriminant Analysis for supervised dimensionality reduction.
%
%   [XLDA, PARAMS] = neuramed.features.applyLDA(X, Y) projects X onto the
%   directions that maximize between-class separability relative to
%   within-class scatter (Fisher's criterion), using labels Y.
%
%   XLDA = neuramed.features.applyLDA(X, [], 'Params', PARAMS) projects
%   new data using previously fitted PARAMS.
%
%   Name-Value Options:
%       'NumComponents' - number of discriminant components
%                          (default min(C-1, D), C = #classes, D = #features)
%       'Params'        - previously fitted LDA params
%
%   Mathematical Definition:
%       Between-class scatter:  S_B = sum_c N_c * (mu_c - mu)(mu_c - mu)'
%       Within-class scatter:   S_W = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)'
%       LDA finds projection directions W solving the generalized
%       eigenproblem:
%           S_B * w = lambda * S_W * w
%       The top (C-1) eigenvectors (ranked by eigenvalue lambda) form the
%       projection matrix W; discriminant scores are Xlda = Xc * W. This
%       maximizes the Fisher criterion J(w) = (w' S_B w) / (w' S_W w).
%
%   Example:
%       [Xtr_l, p] = neuramed.features.applyLDA(Xtrain, Ytrain);
%       Xte_l = neuramed.features.applyLDA(Xtest, [], 'Params', p);
%
%   See also NEURAMED.FEATURES.APPLYPCA

    arguments
        X
        Y
        options.NumComponents = []
        options.Params = []
    end

    EPSILON = 1e-8;
    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);

    if isempty(options.Params)
        Yc = categorical(Y);
        classes = categories(Yc);
        C = numel(classes);
        D = size(Xmat, 2);
        mu = mean(Xmat, 1, 'omitnan');

        Sb = zeros(D, D);
        Sw = zeros(D, D);
        for c = 1:C
            mask = Yc == classes{c};
            Xc_ = Xmat(mask, :);
            Nc = size(Xc_, 1);
            muC = mean(Xc_, 1, 'omitnan');
            diffMu = (muC - mu)';
            Sb = Sb + Nc * (diffMu * diffMu');
            centered = Xc_ - muC;
            Sw = Sw + centered' * centered;
        end
        Sw = Sw + EPSILON * eye(D); % regularize for numerical stability

        [V, D_] = eig(Sb, Sw);
        eigVals = real(diag(D_));
        [eigVals, order] = sort(eigVals, 'descend');
        V = real(V(:, order));

        if isempty(options.NumComponents)
            k = min(C - 1, D);
        else
            k = min(options.NumComponents, D);
        end
        k = max(k, 1);

        params = struct('Mean', mu, 'W', V(:, 1:k), 'EigenValues', eigVals(1:k));
    else
        params = options.Params;
        mu = params.Mean;
    end

    Xlda = (Xmat - mu) * params.W;
end
