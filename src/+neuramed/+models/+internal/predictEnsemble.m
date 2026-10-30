function [predictions, probabilities, scores] = predictEnsemble(model, Xmat)
%PREDICTENSEMBLE (internal) Dispatch prediction for Ensemble-type models.
    ensembleData = model.Underlying;
    baseModels = ensembleData.BaseModels;

    switch ensembleData.CombineMethod
        case 'stacking'
            metaFeatures = neuramed.models.internal.buildMetaFeatures(baseModels, Xmat);
            [predictions, probabilities] = predict(ensembleData.MetaModel, metaFeatures);
            scores = probabilities;

        otherwise % softvote or majorityvote
            allProbs = cell(1, numel(baseModels));
            allPreds = cell(1, numel(baseModels));
            for i = 1:numel(baseModels)
                [p, prob] = neuramed.models.predictModel(baseModels{i}, Xmat);
                allPreds{i} = p;
                allProbs{i} = prob;
            end

            if strcmp(ensembleData.CombineMethod, 'softvote')
                stacked = cat(3, allProbs{:});
                probabilities = mean(stacked, 3);
                classNames = baseModels{1}.Underlying.ClassNames;
                [~, idx] = max(probabilities, [], 2);
                predictions = classNames(idx);
                scores = probabilities;
            else % majorityvote
                predMat = [allPreds{:}];
                predictions = mode(predMat, 2);
                probabilities = [];
                scores = [];
            end
    end
end
