function net = trainDLModel(layers, trainData, trainingOpts)
%TRAINDLMODEL Train a CNN/deep network with standardized logging.
%
%   NET = neuramed.deeplearning.trainDLModel(LAYERS, TRAINDATA, OPTS)
%   trains LAYERS (from buildCNN or transferLearningModel) on TRAINDATA
%   (imageDatastore or augmentedImageDatastore) using trainingOptions
%   OPTS (from neuramed.deeplearning.trainingUtilities), and logs
%   performance via neuramed.utils.logPerformance.
%
%   Example:
%       layers = neuramed.deeplearning.buildCNN([128 128 3], 2);
%       opts = neuramed.deeplearning.trainingUtilities('MaxEpochs', 15);
%       net = neuramed.deeplearning.trainDLModel(layers, augTrain, opts);
%
%   See also NEURAMED.DEEPLEARNING.BUILDCNN, NEURAMED.DEEPLEARNING.TRAININGUTILITIES

    startTime = tic;
    [net, trainInfo] = trainNetwork(trainData, layers, trainingOpts);
    elapsed = toc(startTime);

    neuramed.utils.logPerformance('DeepLearningTraining', struct( ...
        'ElapsedSeconds', elapsed, ...
        'FinalTrainingLoss', trainInfo.TrainingLoss(end), ...
        'FinalTrainingAccuracy', trainInfo.TrainingAccuracy(end)));
end
