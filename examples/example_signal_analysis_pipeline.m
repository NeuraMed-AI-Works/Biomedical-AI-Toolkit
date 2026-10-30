%% EXAMPLE_SIGNAL_ANALYSIS_PIPELINE
% Demonstrates feature engineering (PCA/LDA/feature selection) and model
% comparison on biomedical signal-derived features (e.g. ECG/EEG
% handcrafted feature vectors), using the Neura-Med AI toolkit.

run(fullfile(fileparts(mfilename('fullpath')), '..', 'startup.m'));

%% 1. Synthesize a higher-dimensional feature matrix (e.g. 30 handcrafted
% signal features per observation, 3 diagnostic classes)
rng(7);
N = 300; D = 30;
classLabels = categorical(randi([1 3], N, 1), [1 2 3], {'Normal','Arrhythmia','Ischemia'});

X = randn(N, D);
% Inject class-discriminative signal into first 5 features
catNames = categories(classLabels);
for c = 1:3
    mask = classLabels == catNames{c};
    X(mask, 1:5) = X(mask, 1:5) + (c-2)*1.5;
end

data = struct('X', array2table(X), 'Y', classLabels, 'Type', 'tabular', ...
    'Source', 'synthetic_signal_features', 'Metadata', struct());

%% 2. Split
[trainData, testData] = neuramed.dataset.splitDataset(data, 'TrainFraction', 0.7);

%% 3. Rank & select top features
ranking = neuramed.features.rankFeatures(trainData.X, trainData.Y, 'Method', 'fscore');
disp('Top 10 ranked features:');
disp(ranking(1:10, :));

[Xtrain_sel, selNames] = neuramed.features.selectFeatures(trainData.X, trainData.Y, 'NumFeatures', 10, 'Method', 'fscore');
Xtest_mat = testData.X{:, selNames};
Xtest_sel = array2table(Xtest_mat, 'VariableNames', selNames);

%% 4. Dimensionality reduction with PCA (fit on train only)
[Xtrain_pca, pcaParams] = neuramed.features.applyPCA(Xtrain_sel, 'VarianceExplained', 0.95);
Xtest_pca = neuramed.features.applyPCA(Xtest_sel, 'Params', pcaParams);
fprintf('PCA reduced %d selected features to %d components.\n', width(Xtrain_sel), pcaParams.NumComponents);

%% 5. Supervised reduction with LDA for comparison
[Xtrain_lda, ldaParams] = neuramed.features.applyLDA(Xtrain_sel, trainData.Y);
Xtest_lda = neuramed.features.applyLDA(Xtest_sel, [], 'Params', ldaParams);

%% 6. Train & compare models on PCA-reduced features
models = struct();
models.Tree = neuramed.models.trainDecisionTree(Xtrain_pca, trainData.Y);
models.KNN  = neuramed.models.trainKNN(Xtrain_pca, trainData.Y, 'NumNeighbors', 7);
models.NB   = neuramed.models.trainNaiveBayes(Xtrain_pca, trainData.Y);
models.Ensemble = neuramed.models.trainEnsemble(Xtrain_pca, trainData.Y, ...
    'BaseLearners', {'tree','knn','logistic'}, 'CombineMethod', 'softvote');

modelNames = fieldnames(models);
fprintf('\n%-12s %-10s %-10s\n', 'Model', 'Accuracy', 'MacroF1');
for i = 1:numel(modelNames)
    m = models.(modelNames{i});
    pred = neuramed.models.predictModel(m, Xtest_pca);
    rpt = neuramed.evaluation.generateEvaluationReport(testData.Y, pred);
    fprintf('%-12s %-10.3f %-10.3f\n', modelNames{i}, rpt.Accuracy, rpt.MacroF1);
end
