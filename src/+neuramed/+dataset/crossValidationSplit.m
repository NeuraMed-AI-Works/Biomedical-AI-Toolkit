function folds = crossValidationSplit(data, options)
%CROSSVALIDATIONSPLIT Generate K-fold (optionally stratified) partitions.
%
%   FOLDS = neuramed.dataset.crossValidationSplit(DATA) returns a 1xK
%   struct array, FOLDS(k).Train / FOLDS(k).Test, using 5-fold stratified
%   cross-validation.
%
%   Name-Value Options:
%       'K'         - number of folds (default 5)
%       'Stratify'  - preserve class proportions per fold (default true)
%       'Seed'      - RNG seed (default 42)
%
%   Mathematical Notes:
%       K-fold CV partitions indices {1..N} into K disjoint blocks
%       F_1..F_K of size ~N/K. For stratified CV, this partitioning is
%       performed independently within each class so that
%       |F_k intersect class_c| / |class_c| is approximately equal
%       across all k. Each fold k uses F_k as the held-out test set and
%       the union of the remaining folds as training data, giving an
%       approximately unbiased estimate of generalization error when
%       averaged over folds:
%           CV_error = (1/K) * sum_k Error(model_k, F_k)
%
%   Example:
%       folds = neuramed.dataset.crossValidationSplit(data, 'K', 10);
%       for k = 1:numel(folds)
%           model = neuramed.models.trainRandomForest(folds(k).Train.X, folds(k).Train.Y);
%       end
%
%   See also NEURAMED.DATASET.SPLITDATASET, NEURAMED.EVALUATION.CROSSVALIDATEMODEL

    arguments
        data
        options.K (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.K, 2)} = 5
        options.Stratify (1,1) logical = true
        options.Seed (1,1) double = 42
    end

    rng(options.Seed);
    [X, Y, isStructInput, dataType] = neuramed.dataset.internal.unpack(data);
    n = neuramed.dataset.internal.numObservations(X, dataType);
    K = options.K;

    foldAssignment = zeros(n, 1);

    if options.Stratify && ~isempty(Y)
        classes = unique(Y);
        for c = 1:numel(classes)
            idx = find(Y == classes(c));
            idx = idx(randperm(numel(idx)));
            cvIdx = mod((0:numel(idx)-1)', K) + 1;
            foldAssignment(idx) = cvIdx;
        end
    else
        idx = randperm(n)';
        cvIdx = mod((0:n-1)', K) + 1;
        foldAssignment(idx) = cvIdx;
    end

    folds = repmat(struct('Train', [], 'Test', []), 1, K);
    for k = 1:K
        testIdx = find(foldAssignment == k);
        trainIdx = find(foldAssignment ~= k);
        folds(k).Train = neuramed.dataset.internal.repack(data, X, Y, trainIdx, isStructInput, dataType);
        folds(k).Test = neuramed.dataset.internal.repack(data, X, Y, testIdx, isStructInput, dataType);
    end
end
