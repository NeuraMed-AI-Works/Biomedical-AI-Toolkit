%% EXAMPLE_DISEASE_CLASSIFICATION
% End-to-end tabular disease classification pipeline using the
% Neura-Med AI Biomedical AI Toolkit: load -> validate -> split ->
% engineer features -> train multiple models -> evaluate -> clinical
% decision support -> save.
%
% This example synthesizes a toy biomarker dataset so it runs with no
% external files required. Replace the synthetic-data block with
% neuramed.dataset.loadDataset('your_file.csv', 'LabelColumn', 'Diagnosis')
% for real data.

run(fullfile(fileparts(mfilename('fullpath')), '..', 'startup.m'));

%% 1. Load / synthesize data
rng(1);
N = 400;
age = randi([30 80], N, 1);
bmi = 18 + 15*rand(N, 1);
biomarkerA = randn(N, 1);
biomarkerB = randn(N, 1);
riskScore = 0.03*(age-55) + 0.08*(bmi-25) + 0.6*biomarkerA - 0.3*biomarkerB;
prob = 1 ./ (1 + exp(-riskScore));
label = categorical(rand(N,1) < prob, [false true], {'Negative','Positive'});

T = table(age, bmi, biomarkerA, biomarkerB, 'VariableNames', {'Age','BMI','BiomarkerA','BiomarkerB'});
data = struct('X', T, 'Y', label, 'Type', 'tabular', 'Source', 'synthetic', 'Metadata', struct());
data.Metadata = neuramed.dataset.manageMetadata('create', data);

%% 2. Validate data quality
report = neuramed.dataset.validateData(data);
fprintf('Data valid: %d | Issues found: %d\n', report.IsValid, numel(report.Issues));

%% 3. Split into train/test
[trainData, testData] = neuramed.dataset.splitDataset(data, 'TrainFraction', 0.75);

%% 4. Feature engineering: standardize (fit on train only)
[Xtrain_s, scaleParams] = neuramed.features.standardizeFeatures(trainData.X);
Xtest_s = neuramed.features.standardizeFeatures(testData.X, 'Params', scaleParams);

%% 5. Train several models
rfModel  = neuramed.models.trainRandomForest(Xtrain_s, trainData.Y, 'NumTrees', 100);
svmModel = neuramed.models.trainSVM(Xtrain_s, trainData.Y, 'KernelFunction', 'rbf');
xgbModel = neuramed.models.trainXGBoost(Xtrain_s, trainData.Y, 'NumTrees', 100);

%% 6. Evaluate
[predRF, probRF] = neuramed.models.predictModel(rfModel, Xtest_s);
reportRF = neuramed.evaluation.generateEvaluationReport(testData.Y, predRF, probRF);

fprintf('\n=== Random Forest Evaluation ===\n');
fprintf('Accuracy : %.3f\n', reportRF.Accuracy);
fprintf('Macro F1 : %.3f\n', reportRF.MacroF1);
fprintf('AUC      : %.3f\n', reportRF.AUC);
disp(reportRF.SensitivitySpecificity);

%% 7. Cross-validate for a more robust estimate
cvResults = neuramed.evaluation.crossValidateModel(@neuramed.models.trainRandomForest, data, 'K', 5);
fprintf('\n5-Fold CV Accuracy: %.3f +/- %.3f\n', cvResults.MeanAccuracy, cvResults.StdAccuracy);

%% 8. Clinical decision support on a new patient
newPatient = table(62, 29.4, 1.2, -0.4, 'VariableNames', {'Age','BMI','BiomarkerA','BiomarkerB'});
newPatient_s = neuramed.features.standardizeFeatures(newPatient, 'Params', scaleParams);

clinicalResult = neuramed.clinical.predictDisease(rfModel, newPatient_s);
recommendation = neuramed.clinical.generateClinicalRecommendation(clinicalResult(1));

fprintf('\n=== Clinical Decision Support ===\n');
disp(recommendation.Summary);
disp(recommendation.SuggestedAction);
disp(recommendation.Disclaimer);

%% 9. Save the model
outDir = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~isfolder(outDir), mkdir(outDir); end
neuramed.utils.saveModel(rfModel, fullfile(outDir, 'random_forest_disease_model.mat'), 'Overwrite', true);
fprintf('\nModel saved to %s\n', fullfile(outDir, 'random_forest_disease_model.mat'));
