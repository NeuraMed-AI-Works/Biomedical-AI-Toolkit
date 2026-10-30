function results = batchPredict(model, dataSource, options)
%BATCHPREDICT Run prediction over a large dataset in memory-safe batches.
%
%   RESULTS = neuramed.utils.batchPredict(MODEL, X) predicts over all of
%   X, chunked internally, and concatenates results. Chunking matters for
%   large clinical registries (e.g. hundreds of thousands of records)
%   that should not be materialized as one giant intermediate array.
%
%   Name-Value Options:
%       'BatchSize'   - rows per batch (default 1000)
%       'ShowProgress'- logical, print progress (default true)
%
%   Output RESULTS fields:
%       .Predictions, .Probabilities, .NumProcessed, .ElapsedSeconds
%
%   Example:
%       results = neuramed.utils.batchPredict(model, largeX, 'BatchSize', 5000);
%
%   See also NEURAMED.MODELS.PREDICTMODEL

    arguments
        model (1,1) struct
        dataSource
        options.BatchSize (1,1) double {mustBePositive, mustBeInteger} = 1000
        options.ShowProgress (1,1) logical = true
    end

    [Xmat, isTable, varNames] = neuramed.features.internal.toMatrix(dataSource);
    n = size(Xmat, 1);
    numBatches = ceil(n / options.BatchSize);

    allPreds = cell(numBatches, 1);
    allProbs = cell(numBatches, 1);
    startTime = tic;

    for b = 1:numBatches
        idxStart = (b-1) * options.BatchSize + 1;
        idxEnd = min(b * options.BatchSize, n);
        batchX = neuramed.features.internal.fromMatrix(Xmat(idxStart:idxEnd, :), isTable, varNames, dataSource);

        [p, prob] = neuramed.models.predictModel(model, batchX);
        allPreds{b} = p;
        allProbs{b} = prob;

        if options.ShowProgress
            fprintf('[neuramed] batchPredict: processed %d/%d observations (batch %d/%d)\n', ...
                idxEnd, n, b, numBatches);
        end
    end

    results = struct();
    results.Predictions = vertcat(allPreds{:});
    results.Probabilities = vertcat(allProbs{:});
    results.NumProcessed = n;
    results.ElapsedSeconds = toc(startTime);
end
