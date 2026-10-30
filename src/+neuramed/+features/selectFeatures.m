function [Xsel, selectedNames, idx] = selectFeatures(X, Y, options)
%SELECTFEATURES Select the top-K most relevant features.
%
%   [XSEL, NAMES, IDX] = neuramed.features.selectFeatures(X, Y) ranks
%   features via neuramed.features.rankFeatures and returns the top-K
%   subset (default K=10 or all features if fewer).
%
%   Name-Value Options:
%       'NumFeatures' - K, number of features to keep (default 10)
%       'Method'      - ranking method, forwarded to rankFeatures (default 'mrmr')
%
%   Example:
%       [Xsel, names] = neuramed.features.selectFeatures(X, Y, 'NumFeatures', 15);
%
%   See also NEURAMED.FEATURES.RANKFEATURES

    arguments
        X
        Y
        options.NumFeatures (1,1) double {mustBePositive, mustBeInteger} = 10
        options.Method (1,:) char = 'mrmr'
    end

    ranking = neuramed.features.rankFeatures(X, Y, 'Method', options.Method);
    K = min(options.NumFeatures, height(ranking));
    selectedNames = ranking.Feature(1:K);

    [Xmat, isTable, varNames] = neuramed.features.internal.toMatrix(X);
    if isempty(varNames)
        varNames = arrayfun(@(i) sprintf('Feature%d', i), 1:size(Xmat,2), 'UniformOutput', false);
    end
    [~, idx] = ismember(selectedNames, varNames);

    Xsub = Xmat(:, idx);
    Xsel = neuramed.features.internal.fromMatrix(Xsub, isTable, selectedNames, X);
end
