function [trainData, testData, valData] = splitDataset(data, options)
%SPLITDATASET Partition a dataset into train/test/validation subsets.
%
%   [TRAIN, TEST] = neuramed.dataset.splitDataset(DATA) splits DATA
%   (struct from loadDataset, or a table/matrix + label pair) into a
%   70/30 stratified train/test split.
%
%   [TRAIN, TEST, VAL] = neuramed.dataset.splitDataset(DATA, 'ValFraction', 0.15)
%   additionally carves out a validation set.
%
%   Name-Value Options:
%       'TrainFraction' (default 0.70)
%       'ValFraction'   (default 0)
%       'TestFraction'  (default computed as 1 - Train - Val)
%       'Stratify'      (default true)  - preserve class proportions
%       'Seed'          (default 42)    - RNG seed for reproducibility
%
%   Mathematical Notes:
%       Stratified sampling partitions indices per class c into
%       floor(p * n_c) samples for each split, ensuring
%       P(class = c | split) approx P(class = c) for every split,
%       which controls for class-imbalance bias in downstream evaluation.
%
%   Example:
%       [train, test] = neuramed.dataset.splitDataset(data, 'TrainFraction', 0.8);
%
%   See also NEURAMED.DATASET.CROSSVALIDATIONSPLIT

    arguments
        data
        options.TrainFraction (1,1) double {mustBeInRange(options.TrainFraction, 0, 1)} = 0.70
        options.ValFraction (1,1) double {mustBeInRange(options.ValFraction, 0, 1)} = 0
        options.TestFraction double = []
        options.Stratify (1,1) logical = true
        options.Seed (1,1) double = 42
    end

    if isempty(options.TestFraction)
        options.TestFraction = 1 - options.TrainFraction - options.ValFraction;
    end
    total = options.TrainFraction + options.ValFraction + options.TestFraction;
    if options.TestFraction < -1e-12 || options.TestFraction > 1 + 1e-12 || abs(total - 1) > 1e-6
        error('neuramed:dataset:splitDataset:FractionsInvalid', ...
            'TrainFraction + ValFraction + TestFraction must sum to 1 (got %.4f).', total);
    end

    rng(options.Seed);

    [X, Y, isStruct, origType] = neuramed.dataset.internal.unpack(data);
    n = neuramed.dataset.internal.numObservations(X, origType);

    if options.Stratify && ~isempty(Y)
        classes = unique(Y);
        trainIdx = []; valIdx = []; testIdx = [];
        for i = 1:numel(classes)
            idx = find(Y == classes(i));
            idx = idx(randperm(numel(idx)));
            nC = numel(idx);
            nTrain = floor(options.TrainFraction * nC);
            nVal = floor(options.ValFraction * nC);
            trainIdx = [trainIdx; idx(1:nTrain)]; %#ok<AGROW>
            valIdx = [valIdx; idx(nTrain+1:nTrain+nVal)]; %#ok<AGROW>
            testIdx = [testIdx; idx(nTrain+nVal+1:end)]; %#ok<AGROW>
        end
    else
        idx = randperm(n);
        nTrain = floor(options.TrainFraction * n);
        nVal = floor(options.ValFraction * n);
        trainIdx = idx(1:nTrain)';
        valIdx = idx(nTrain+1:nTrain+nVal)';
        testIdx = idx(nTrain+nVal+1:end)';
    end

    trainData = neuramed.dataset.internal.repack(data, X, Y, trainIdx, isStruct, origType);
    testData  = neuramed.dataset.internal.repack(data, X, Y, testIdx, isStruct, origType);
    if options.ValFraction > 0
        valData = neuramed.dataset.internal.repack(data, X, Y, valIdx, isStruct, origType);
    else
        valData = [];
    end
end
