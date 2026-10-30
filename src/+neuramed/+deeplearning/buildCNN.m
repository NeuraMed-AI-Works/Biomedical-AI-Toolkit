function layers = buildCNN(inputSize, numClasses, options)
%BUILDCNN Construct a configurable CNN architecture for medical images.
%
%   LAYERS = neuramed.deeplearning.buildCNN(INPUTSIZE, NUMCLASSES) builds
%   a layer graph, e.g. INPUTSIZE = [224 224 3], NUMCLASSES = 2.
%
%   Name-Value Options:
%       'NumBlocks'     - number of conv-pool blocks (default 4)
%       'InitialFilters'- filters in first block, doubling each block (default 32)
%       'DropoutRate'   - dropout before the classifier head (default 0.4)
%       'FilterSize'    - conv kernel size (default 3)
%
%   Mathematical Notes:
%       Each convolutional layer computes a feature map via discrete
%       convolution: y[i,j,k] = sum_{m,n,c} W[m,n,c,k] * x[i+m,j+n,c] + b_k
%       followed by a ReLU nonlinearity f(z) = max(0, z), which injects
%       non-linearity for hierarchical feature learning. Batch
%       normalization standardizes activations per mini-batch
%       (z' = (z - batch_mean) / sqrt(batch_var + eps)) to stabilize and
%       accelerate training. Max-pooling downsamples via
%       y = max over a local window, providing translation-invariance
%       and reducing spatial resolution/computation. The final dropout
%       layer randomly zeroes activations with probability p during
%       training as a stochastic regularizer, approximating an ensemble
%       average over 2^n thinned sub-networks (Bayesian-approximation
%       interpretation of dropout).
%
%   Example:
%       layers = neuramed.deeplearning.buildCNN([128 128 1], 3, 'NumBlocks', 3);
%
%   See also NEURAMED.DEEPLEARNING.TRANSFERLEARNINGMODEL, NEURAMED.DEEPLEARNING.TRAINDLMODEL

    arguments
        inputSize (1,:) double
        numClasses (1,1) double {mustBePositive, mustBeInteger}
        options.NumBlocks (1,1) double {mustBePositive, mustBeInteger} = 4
        options.InitialFilters (1,1) double {mustBePositive, mustBeInteger} = 32
        options.DropoutRate (1,1) double {mustBeInRange(options.DropoutRate, 0, 1)} = 0.4
        options.FilterSize (1,1) double {mustBePositive, mustBeInteger} = 3
    end

    layerList = imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zscore');

    numFilters = options.InitialFilters;
    for b = 1:options.NumBlocks
        layerList = [layerList
            convolution2dLayer(options.FilterSize, numFilters, 'Padding', 'same', 'Name', sprintf('conv%d', b))
            batchNormalizationLayer('Name', sprintf('bn%d', b))
            reluLayer('Name', sprintf('relu%d', b))
            maxPooling2dLayer(2, 'Stride', 2, 'Name', sprintf('pool%d', b))]; %#ok<AGROW>
        numFilters = numFilters * 2;
    end

    layerList = [layerList
        dropoutLayer(options.DropoutRate, 'Name', 'dropout')
        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')];

    layers = layerList;
end
