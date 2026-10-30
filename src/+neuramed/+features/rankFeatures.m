function ranking = rankFeatures(X, Y, options)
%RANKFEATURES Rank features by univariate relevance to the target label.
%
%   RANKING = neuramed.features.rankFeatures(X, Y) scores each feature in
%   X against label Y and returns a table sorted by descending relevance.
%
%   Name-Value Options:
%       'Method' - 'mrmr' | 'fscore' | 'chi2' | 'relieff' (default 'mrmr')
%       'NumNeighbors' - for 'relieff' (default 10)
%
%   Mathematical Notes:
%       'fscore' (ANOVA F-test) per feature j:
%           F_j = (Between-class variance) / (Within-class variance)
%       'chi2' tests independence between binned feature values and
%       class labels via the chi-squared statistic.
%       'mrmr' (Minimum Redundancy Maximum Relevance) selects features
%       maximizing mutual information with Y while minimizing mutual
%       information with already-selected features:
%           score_j = I(x_j; Y) - (1/|S|) * sum_{s in S} I(x_j; x_s)
%       'relieff' estimates feature quality from nearest-neighbor
%       margins in feature space, robust to feature interactions.
%
%   Example:
%       ranking = neuramed.features.rankFeatures(X, Y, 'Method', 'mrmr');
%       topFeatures = ranking.Feature(1:10);
%
%   See also NEURAMED.FEATURES.SELECTFEATURES

    arguments
        X
        Y
        options.Method (1,:) char {mustBeMember(options.Method, {'mrmr','fscore','chi2','relieff'})} = 'mrmr'
        options.NumNeighbors (1,1) double {mustBePositive, mustBeInteger} = 10
    end

    [Xmat, ~, varNames] = neuramed.features.internal.toMatrix(X);
    if isempty(varNames)
        varNames = arrayfun(@(i) sprintf('Feature%d', i), 1:size(Xmat,2), 'UniformOutput', false);
    end
    Yc = categorical(Y);

    switch options.Method
        case 'mrmr'
            [idx, scores] = fscmrmr(Xmat, Yc);
        case 'fscore'
            [idx, scores] = neuramed.features.internal.fscoreRank(Xmat, Yc);
        case 'chi2'
            [idx, scores] = neuramed.features.internal.chi2Rank(Xmat, Yc);
        case 'relieff'
            [idx, scores] = relieff(Xmat, Yc, options.NumNeighbors);
    end

    ranking = table(varNames(idx)', scores(idx)', (1:numel(idx))', ...
        'VariableNames', {'Feature', 'Score', 'Rank'});
end
