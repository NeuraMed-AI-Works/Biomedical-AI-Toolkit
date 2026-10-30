%% EXAMPLE_MEDICAL_IMAGE_CLASSIFICATION
% Demonstrates the deep-learning module: building a CNN from scratch,
% augmenting a medical image datastore, and training/evaluating.
%
% NOTE: This example expects an image folder structured as:
%   imageFolder/ClassA/*.jpg
%   imageFolder/ClassB/*.jpg
% and requires Deep Learning Toolbox. It is written to run end-to-end
% given such a folder; adjust IMAGE_FOLDER before running.

run(fullfile(fileparts(mfilename('fullpath')), '..', 'startup.m'));

IMAGE_FOLDER = fullfile(fileparts(mfilename('fullpath')), 'sample_images'); % <-- set to your data

if ~isfolder(IMAGE_FOLDER)
    fprintf(['This example expects labeled image subfolders at:\n  %s\n' ...
        'Skipping execution: no sample_images/ folder found in this environment.\n' ...
        'Point IMAGE_FOLDER at your dataset (e.g. chest_xrays/Normal, chest_xrays/Pneumonia) to run.\n'], IMAGE_FOLDER);
    return
end

%% 1. Build labeled image datastore
imds = neuramed.deeplearning.createImageDatastore(IMAGE_FOLDER);
numClasses = numel(categories(imds.Labels));

%% 2. Split (stratified) into train/val/test using the dataset module
imgData = struct('X', imds, 'Y', imds.Labels, 'Type', 'image', 'Source', IMAGE_FOLDER, 'Metadata', struct());
[trainD, testD, valD] = neuramed.dataset.splitDataset(imgData, 'TrainFraction', 0.7, 'ValFraction', 0.15);

%% 3. On-the-fly augmentation
inputSize = [128 128 3];
augTrain = neuramed.deeplearning.augmentImageData(trainD.X, inputSize);
augVal   = neuramed.deeplearning.augmentImageData(valD.X, inputSize, ...
    'RotationRange', [0 0], 'ScaleRange', [1 1], 'ReflectHorizontal', false); % no aug at val time
augTest  = neuramed.deeplearning.augmentImageData(testD.X, inputSize, ...
    'RotationRange', [0 0], 'ScaleRange', [1 1], 'ReflectHorizontal', false);

%% 4. Build CNN architecture
layers = neuramed.deeplearning.buildCNN(inputSize, numClasses, ...
    'NumBlocks', 3, 'InitialFilters', 16, 'DropoutRate', 0.4);

%% 5. Configure training (GPU auto-detected)
opts = neuramed.deeplearning.trainingUtilities( ...
    'MaxEpochs', 15, 'MiniBatchSize', 16, 'InitialLearnRate', 1e-4, ...
    'ValidationData', augVal);

%% 6. Train
net = neuramed.deeplearning.trainDLModel(layers, augTrain, opts);

%% 7. Evaluate on held-out test set
results = neuramed.deeplearning.evaluateDLModel(net, augTest, testD.Y);
fprintf('\n=== CNN Test Evaluation ===\n');
fprintf('Accuracy: %.3f | Macro F1: %.3f | AUC: %.3f\n', ...
    results.Accuracy, results.MacroF1, results.AUC);

%% 8. Save trained network via the generic model wrapper
wrapped = struct('Underlying', net, 'Type', 'CNN', 'TrainedAt', datetime('now'), ...
    'Options', struct('InputSize', inputSize), 'ToolkitVersion', neuramed.utils.manageVersion('get'));
neuramed.utils.saveModel(wrapped, fullfile(fileparts(mfilename('fullpath')), 'output', 'cnn_model.mat'), 'Overwrite', true);
