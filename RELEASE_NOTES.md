# Release Notes — Neura-Med AI Biomedical AI Toolkit

All notable changes to this toolkit are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-07-10

### Added — Initial Release

**Module 1: Dataset Management** (`neuramed.dataset`)
- `loadDataset` — unified loader for tabular (.csv/.xlsx/.mat), image-folder, and signal-folder data
- `splitDataset` — stratified train/test/validation partitioning
- `validateData` — missing-value, near-constant-feature, duplicate, and class-imbalance checks
- `crossValidationSplit` — stratified K-fold partition generation
- `augmentData` — variance-scaled Gaussian jitter (tabular/signal) and affine augmentation (image)
- `manageMetadata` — dataset provenance tracking with content-hash verification

**Module 2: Feature Engineering** (`neuramed.features`)
- `normalizeFeatures` — min-max scaling with fit/reuse params
- `standardizeFeatures` — z-score standardization with fit/reuse params
- `scaleFeatures` — robust median/IQR scaling
- `rankFeatures` — mRMR, ANOVA F-score, chi-squared, and ReliefF ranking methods
- `selectFeatures` — top-K feature selection built on `rankFeatures`
- `applyPCA` — Principal Component Analysis with variance-explained or fixed-K selection
- `applyLDA` — Linear Discriminant Analysis (Fisher's criterion)
- `reduceDimensionality` — unified PCA/t-SNE dispatch

**Module 3: Machine Learning Models** (`neuramed.models`)
- `trainDecisionTree`, `trainRandomForest`, `trainSVM`, `trainKNN`, `trainNaiveBayes`,
  `trainLogisticRegression`, `trainXGBoost` (via `fitcensemble` gradient boosting — documented
  toolbox-native equivalent, no native MATLAB XGBoost binding exists), `trainEnsemble`
  (soft-vote / majority-vote / stacking)
- `predictModel` — single dispatch point for prediction across every model type

**Module 4: Deep Learning Utilities** (`neuramed.deeplearning`)
- `buildCNN` — configurable conv/batchnorm/relu/pool block stack
- `transferLearningModel` — head replacement for resnet18/resnet50/squeezenet/mobilenetv2
- `createImageDatastore`, `augmentImageData` — labeled datastore construction + on-the-fly augmentation
- `trainingUtilities` — standardized `trainingOptions` with GPU auto-detection
- `trainDLModel`, `evaluateDLModel` — training/evaluation wrappers routed through the toolkit's
  standard logging and reporting
- `checkGPU` — GPU availability/capability detection

**Module 5: Model Evaluation** (`neuramed.evaluation`)
- `computeAccuracy`, `computeConfusionMatrix`, `computePrecisionRecall`,
  `computeSpecificitySensitivity`, `computeF1Score`, `computeROCAUC`
- `crossValidateModel` — K-fold CV wrapper for any `train*` function
- `generateEvaluationReport` — one-call comprehensive evaluation report

**Module 6: Clinical Decision Module** (`neuramed.clinical`)
- `predictDisease` — clinical-facing prediction wrapper
- `computeConfidenceScore`, `estimateProbability` (with predictive entropy)
- `classifyRisk` — configurable risk-tier thresholding
- `generateClinicalRecommendation` — templated recommendation text with mandatory-by-default
  non-diagnostic disclaimer

**Module 7: Utilities** (`neuramed.utils`)
- `saveModel`, `loadModel` — versioned model persistence
- `manageVersion` — semantic version get/set
- `batchPredict` — memory-safe chunked batch prediction
- `logPerformance` — append-only JSON-lines audit logging
- `manageConfig` — JSON configuration load/save/get

**Testing & Documentation**
- `matlab.unittest`-based test suite: `test_dataset_module`, `test_features_module`,
  `test_models_module`, `test_evaluation_module`, `test_clinical_module`, `test_utils_module`,
  runnable via `tests/run_all_tests.m`
- Three end-to-end example pipelines: tabular disease classification, signal feature-engineering
  and model comparison, and medical image CNN classification
- `docs/ARCHITECTURE.md` — design rationale, data contracts, module dependency graph
- `docs/USER_MANUAL.md` — full mathematical treatment of every algorithm in the toolkit
- `.github/workflows/ci.yml` — CI test-suite runner scaffold

### Known Limitations
- `trainXGBoost` uses MATLAB's native gradient-boosting (`fitcensemble`), not the reference
  XGBoost library; results are conceptually aligned (same boosting/loss family) but numerically
  distinct from a `xgboost` Python/R fit.
- `reduceDimensionality`'s `'tsne'` method is transductive and does not support out-of-sample
  `'Params'` reuse; it must be re-run on any new dataset.
- Default risk-tier thresholds in `classifyRisk` are illustrative placeholders and require
  clinical validation (e.g. decision-curve or Youden's-J analysis) before any production use.
- `.github/workflows/ci.yml` assumes a MATLAB-licensed runner (self-hosted or MathWorks-hosted);
  it will not run on a bare `ubuntu-latest` GitHub-hosted runner without a MATLAB license action.

---

## Versioning Policy Going Forward

- **PATCH** (1.0.x): bug fixes, documentation corrections, non-breaking test additions.
- **MINOR** (1.x.0): new functions, new options on existing functions (with backward-compatible
  defaults), new example pipelines.
- **MAJOR** (x.0.0): any change to an existing function's required-argument signature, return
  value shape, or the core dataset/model struct contracts described in `docs/ARCHITECTURE.md`.

Update `VERSION.txt` via `neuramed.utils.manageVersion('set', 'X.Y.Z')` when cutting a release,
and add a corresponding section to this file.
