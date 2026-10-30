function augImds = augmentImageData(imds, outputSize, options)
%AUGMENTIMAGEDATA Wrap an imageDatastore with online augmentation for training.
%
%   AUGIMDS = neuramed.deeplearning.augmentImageData(IMDS, [224 224])
%   returns an augmentedImageDatastore that resizes and augments images
%   on-the-fly during training (memory-efficient; no precomputed copies).
%
%   Name-Value Options:
%       'RotationRange'     - degrees (default [-10 10])
%       'ReflectHorizontal' - logical (default true)
%       'ScaleRange'        - default [0.9 1.1]
%       'TranslationRange'  - pixels, default [-5 5]
%
%   Mathematical Notes:
%       On-the-fly augmentation draws a fresh random affine transform
%       T = Translate(t) * Rotate(theta) * Scale(s) independently for
%       every image at every epoch, so the effective training
%       distribution is a continuous augmentation manifold around each
%       original sample rather than a fixed finite augmented set. This
%       reduces overfitting risk for small biomedical imaging datasets,
%       where collecting more labeled data is expensive.
%
%   Example:
%       augImds = neuramed.deeplearning.augmentImageData(imds, [224 224 3]);
%
%   See also NEURAMED.DEEPLEARNING.CREATEIMAGEDATASTORE

    arguments
        imds
        outputSize (1,:) double
        options.RotationRange (1,2) double = [-10 10]
        options.ReflectHorizontal (1,1) logical = true
        options.ScaleRange (1,2) double = [0.9 1.1]
        options.TranslationRange (1,2) double = [-5 5]
    end

    augmenter = imageDataAugmenter( ...
        'RandRotation', options.RotationRange, ...
        'RandXReflection', options.ReflectHorizontal, ...
        'RandScale', options.ScaleRange, ...
        'RandXTranslation', options.TranslationRange, ...
        'RandYTranslation', options.TranslationRange);

    colorMode = 'none';
if numel(outputSize) >= 3 && outputSize(3) == 3, colorMode = 'gray2rgb'; end
if numel(outputSize) >= 3 && outputSize(3) == 1, colorMode = 'rgb2gray'; end
augImds = augmentedImageDatastore(outputSize(1:2), imds, 'DataAugmentation', augmenter, 'ColorPreprocessing', colorMode);
end
