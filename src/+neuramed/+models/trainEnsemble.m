function model = trainEnsemble(X, Y, options)
%TRAINENSEMBLE Train a stacked/voting ensemble over multiple base learners.
%
%   MODEL = neuramed.models.trainEnsemble(X, Y) trains several base
%   models (decision tree, SVM, KNN, logistic regression by default) and
%   combines their predictions via soft voting or stacking.
%
%   Name-Value Options:
%       'BaseLearners' - cell array of learner name strings, subset of
%                         {'tree','svm','knn','naivebayes','logistic','randomforest'}
%                         (default {'tree','svm','knn','logistic'})
%       'CombineMethod'- 'softvote' | 'majorityvote' | 'stacking' (default 'softvote')
%
%   Mathematical Notes:
%       Soft voting averages posterior class probabilities across M base
%       models:
%           P_ensemble(c|x) = (1/M) * sum_m P_m(c|x)
%           yhat = argmax_c P_ensemble(c|x)
%       This reduces variance relative to any single model when base
%       models make partially uncorrelated errors (bias-variance
%       decomposition: averaging M models with pairwise error
%       correlation rho reduces variance by a factor between 1/M
%       (rho=0) and 1 (rho=1)). 'Stacking' instead trains a
%       meta-classifier (logistic regression) on the base models'
%       predicted probabilities as new features.
%
%   Example:
%       model = neuramed.models.trainEnsemble(Xtrain, Ytrain, ...
%           'BaseLearners', {'tree','randomforest','svm'}, 'CombineMethod', 'softvote');
%
%   See also NEURAMED.MODELS.PREDICTMODEL

    arguments
        X
        Y
        options.BaseLearners cell = {'tree','svm','knn','logistic'}
        options.CombineMethod (1,:) char {mustBeMember(options.CombineMethod, {'softvote','majorityvote','stacking'})} = 'softvote'
    end

    validLearners = {'tree','svm','knn','naivebayes','logistic','randomforest'};
    for i = 1:numel(options.BaseLearners)
        if ~ismember(options.BaseLearners{i}, validLearners)
            error('neuramed:models:trainEnsemble:UnknownLearner', ...
                'Unknown base learner "%s". Valid options: %s', ...
                options.BaseLearners{i}, strjoin(validLearners, ', '));
        end
    end

    baseModels = cell(1, numel(options.BaseLearners));
    for i = 1:numel(options.BaseLearners)
        switch options.BaseLearners{i}
            case 'tree'
                baseModels{i} = neuramed.models.trainDecisionTree(X, Y);
            case 'svm'
                baseModels{i} = neuramed.models.trainSVM(X, Y);
            case 'knn'
                baseModels{i} = neuramed.models.trainKNN(X, Y);
            case 'naivebayes'
                baseModels{i} = neuramed.models.trainNaiveBayes(X, Y);
            case 'logistic'
                baseModels{i} = neuramed.models.trainLogisticRegression(X, Y);
            case 'randomforest'
                baseModels{i} = neuramed.models.trainRandomForest(X, Y);
        end
    end

    ensembleData = struct();
    ensembleData.BaseModels = baseModels;
    ensembleData.CombineMethod = options.CombineMethod;

    if strcmp(options.CombineMethod, 'stacking')
        [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
        Yc = categorical(Y);
        metaFeatures = neuramed.models.internal.buildMetaFeatures(baseModels, Xmat);
        metaModel = fitclinear(metaFeatures, Yc, 'Learner', 'logistic');
        ensembleData.MetaModel = metaModel;
    end

    model = neuramed.models.internal.wrapModel(ensembleData, 'Ensemble', options);
end
