function [idx, scores] = fscoreRank(X, Y)
%FSCORERANK (internal) One-way ANOVA F-score per feature.
    classes = categories(Y);
    nFeatures = size(X, 2);
    scores = zeros(1, nFeatures);
    grandMean = mean(X, 1, 'omitnan');
    for j = 1:nFeatures
        ssBetween = 0; ssWithin = 0;
        for c = 1:numel(classes)
            mask = Y == classes{c};
            xc = X(mask, j);
            nc = sum(~isnan(xc));
            if nc < 2, continue; end
            ssBetween = ssBetween + nc * (mean(xc, 'omitnan') - grandMean(j))^2;
            ssWithin = ssWithin + sum((xc - mean(xc, 'omitnan')).^2, 'omitnan');
        end
        scores(j) = ssBetween / max(ssWithin, 1e-12);
    end
    [~, idx] = sort(scores, 'descend');
end
