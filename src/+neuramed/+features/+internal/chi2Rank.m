function [idx, scores] = chi2Rank(X, Y)
%CHI2RANK (internal) Chi-squared independence test per binned feature.
    classes = categories(Y);
    nFeatures = size(X, 2);
    nBins = 5;
    scores = zeros(1, nFeatures);
    for j = 1:nFeatures
        col = X(:, j);
        edges = linspace(min(col), max(col) + eps, nBins + 1);
        binIdx = discretize(col, edges);
        observed = zeros(nBins, numel(classes));
        for c = 1:numel(classes)
            mask = Y == classes{c};
            for b = 1:nBins
                observed(b, c) = sum(binIdx(mask) == b);
            end
        end
        rowTotals = sum(observed, 2);
        colTotals = sum(observed, 1);
        total = sum(observed(:));
        expected = (rowTotals * colTotals) / max(total, 1);
        chi2 = sum((observed(:) - expected(:)).^2 ./ max(expected(:), 1e-12));
        scores(j) = chi2;
    end
    [~, idx] = sort(scores, 'descend');
end
