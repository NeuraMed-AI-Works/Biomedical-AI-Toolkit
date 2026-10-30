function model = trainRandomForest(X, Y, options)
%TRAINRANDOMFOREST Train a Random Forest (bagged decision tree ensemble).
%
%   MODEL = neuramed.models.trainRandomForest(X, Y) fits an ensemble of
%   decision trees via bootstrap aggregation with random feature subsets.
%
%   Name-Value Options:
%       'NumTrees'         - number of trees (default 100)
%       'MinLeafSize'      - minimum leaf size (default 1)
%       'NumFeaturesToSample' - features considered per split
%                               (default 'sqrt' -> round(sqrt(D)))
%
%   Mathematical Notes:
%       For b = 1..B trees: draw a bootstrap sample D_b of size N from
%       training data (sampling with replacement, so each D_b holds
%       ~63.2% unique observations), then grow a tree where, at each
%       split, only a random subset of m = sqrt(D) features are
%       considered. Prediction aggregates across trees by majority vote:
%           yhat(x) = mode_b { Tree_b(x) }
%       Out-of-bag (OOB) samples (the ~36.8% not drawn in D_b) give a
%       built-in unbiased error estimate without a held-out set.
%       Variance reduction: Var(mean of B trees) ~ (1/B) Var(single tree)
%       under independence; correlated trees (from correlated features)
%       reduce this benefit, which is why feature subsampling matters.
%
%   Example:
%       model = neuramed.models.trainRandomForest(Xtrain, Ytrain, 'NumTrees', 200);
%
%   See also NEURAMED.MODELS.TRAINDECISIONTREE, NEURAMED.MODELS.TRAINENSEMBLE

    arguments
        X
        Y
        options.NumTrees (1,1) double {mustBePositive, mustBeInteger} = 100
        options.MinLeafSize (1,1) double {mustBePositive, mustBeInteger} = 1
        options.NumFeaturesToSample = 'sqrt'
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
    Yc = categorical(Y);

    % templateTree does not accept 'sqrt'; convert to a positive integer.
    D = size(Xmat, 2);
    nfs = options.NumFeaturesToSample;
    if (ischar(nfs) || isstring(nfs)) && strcmpi(nfs, 'sqrt')
        nfs = max(1, round(sqrt(D)));
    elseif (ischar(nfs) || isstring(nfs)) && strcmpi(nfs, 'all')
        nfs = 'all';
    else
        nfs = min(D, max(1, round(double(nfs))));
    end

    template = templateTree('MinLeafSize', options.MinLeafSize, ...
        'NumVariablesToSample', nfs);

    forest = fitcensemble(Xmat, Yc, ...
        'Method', 'Bag', ...
        'NumLearningCycles', options.NumTrees, ...
        'Learners', template);

    model = neuramed.models.internal.wrapModel(forest, 'RandomForest', options);
end
