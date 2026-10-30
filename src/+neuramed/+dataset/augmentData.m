function augmented = augmentData(data, options)
%AUGMENTDATA Apply domain-appropriate augmentation to biomedical data.
%
%   AUGMENTED = neuramed.dataset.augmentData(DATA) augments tabular,
%   signal, or image data depending on DATA.Type.
%
%   Name-Value Options (Tabular / Signal):
%       'NoiseStd'      - Gaussian noise std as fraction of feature std (default 0.02)
%       'Multiplier'    - number of synthetic samples per original (default 1)
%       'Seed'          - RNG seed (default 42)
%
%   Name-Value Options (Image):
%       'RotationRange'   - degrees, e.g. [-15 15] (default [-15 15])
%       'ScaleRange'      - e.g. [0.9 1.1] (default [0.9 1.1])
%       'ReflectHorizontal' - logical (default true)
%
%   Mathematical Notes:
%       Tabular/Signal jitter augmentation draws x' = x + eps,
%       eps ~ N(0, (sigma_j * NoiseStd)^2) per feature j, i.e. additive
%       Gaussian noise scaled to each feature's own variability. This is
%       a variance-preserving perturbation that keeps synthetic samples
%       close to the original data manifold, which is preferable to
%       fixed-scale noise for heterogeneous biomedical features (e.g.
%       age in years vs. a normalized biomarker in [0,1]).
%
%       Image augmentation uses affine transforms T = R(theta) * S(s),
%       composing rotation and isotropic scaling, implemented via
%       imageDataAugmenter/augmentedImageDatastore.
%
%   Example:
%       augmented = neuramed.dataset.augmentData(data, 'Multiplier', 2);
%
%   See also NEURAMED.DATASET.LOADDATASET

    arguments
        data
        options.NoiseStd (1,1) double {mustBePositive} = 0.02
        options.Multiplier (1,1) double {mustBeInteger, mustBePositive} = 1
        options.Seed (1,1) double = 42
        options.RotationRange (1,2) double = [-15 15]
        options.ScaleRange (1,2) double = [0.9 1.1]
        options.ReflectHorizontal (1,1) logical = true
    end

    rng(options.Seed);

    if isstruct(data) && isfield(data, 'Type') && strcmp(data.Type, 'image')
        augmenter = imageDataAugmenter( ...
            'RandRotation', options.RotationRange, ...
            'RandScale', options.ScaleRange, ...
            'RandXReflection', options.ReflectHorizontal);
        imgSize = size(readimage(data.X, 1));
        augmented = data;
        augmented.X = augmentedImageDatastore(imgSize(1:2), data.X, 'DataAugmentation', augmenter);
        return
    end

    [X, Y, isStructInput, dataType] = neuramed.dataset.internal.unpack(data);

    if strcmp(dataType, 'signal')
        n = numel(X);
        newX = cell(n * options.Multiplier, 1);
        newY = [];
        if ~isempty(Y)
            newY = repmat(Y(:), options.Multiplier, 1);
        end
        k = 1;
        for m = 1:options.Multiplier
            for i = 1:n
                sig = X{i};
                sigma = std(sig(:));
                newX{k} = sig + options.NoiseStd * sigma * randn(size(sig));
                k = k + 1;
            end
        end
        augmented = neuramed.dataset.internal.repack(data, newX, newY, (1:numel(newX))', isStructInput, dataType);
        return
    end

    % Tabular case
    numericVars = varfun(@isnumeric, X, 'OutputFormat', 'uniform');
    Xnum = X{:, numericVars};
    sigma = std(Xnum, 0, 1, 'omitnan');

    repX = repmat(X, options.Multiplier, 1);
    XnumRep = repmat(Xnum, options.Multiplier, 1);
    noise = randn(size(XnumRep)) .* (sigma * options.NoiseStd);
    repX{:, numericVars} = XnumRep + noise;

    newY = [];
    if ~isempty(Y)
        newY = repmat(Y(:), options.Multiplier, 1);
    end

    augmented = neuramed.dataset.internal.repack(data, repX, newY, (1:height(repX))', isStructInput, dataType);
end
