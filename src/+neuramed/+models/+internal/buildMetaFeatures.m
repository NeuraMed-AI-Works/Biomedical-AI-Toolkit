function metaX = buildMetaFeatures(baseModels, Xmat)
%BUILDMETAFEATURES (internal) Build stacking meta-features from base model probabilities.
    probsList = cell(1, numel(baseModels));
    for i = 1:numel(baseModels)
        [~, probs] = neuramed.models.predictModel(baseModels{i}, Xmat);
        probsList{i} = probs;
    end
    metaX = [probsList{:}];
end
