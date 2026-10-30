# Neura-Med AI Biomedical AI Toolkit

**A modular, production-grade MATLAB toolbox for biomedical machine learning, deep learning, and clinical decision support — built for reuse across Neura-Med AI products.**

[![Toolkit Version](https://img.shields.io/badge/version-1.0.0-blue)](RELEASE_NOTES.md)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2022b%2B-orange)](#requirements)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](LICENSE)

---

## Overview

The Neura-Med AI Biomedical AI Toolkit is a namespaced MATLAB package (`+neuramed`) providing reusable, tested, documented building blocks for:

- **Medical Image Analysis** — CNNs, transfer learning, augmented datastores
- **Biomedical Signal Analysis** — feature engineering pipelines for ECG/EEG/PPG-derived features
- **Disease Classification** — 8 classical ML algorithms behind one consistent training/prediction interface
- **Predictive Analytics** — cross-validation, ranking, dimensionality reduction
- **Clinical Decision Support** — confidence scoring, risk tiering, templated recommendations with mandatory non-diagnostic disclaimers

Every function is self-contained, unit-tested, documented with a mathematical/computational explanation of the underlying algorithm, and designed to be dropped into any future Neura-Med AI product without modification.

## Why a namespaced package?

All functionality lives under the `neuramed` MATLAB package (the `+neuramed` folder and its sub-packages `+dataset`, `+features`, `+models`, `+deeplearning`, `+evaluation`, `+clinical`, `+utils`). This:

- Prevents function-name collisions with other toolboxes or MATLAB built-ins
- Makes call sites self-documenting (`neuramed.models.trainRandomForest(...)`)
- Lets internal helpers (`+internal` sub-packages) stay hidden from the public API surface
- Mirrors how MathWorks structures its own toolboxes (`matlab.unittest.*`, etc.)

## Quick Start

```matlab
% 1. Initialize the toolkit (adds src/ to path)
run('NeuraMedAI-Toolkit/startup.m');

% 2. Load and validate data
data = neuramed.dataset.loadDataset('patients.csv', 'LabelColumn', 'Diagnosis');
report = neuramed.dataset.validateData(data);

% 3. Split, engineer features (fit-on-train-only)
[trainData, testData] = neuramed.dataset.splitDataset(data, 'TrainFraction', 0.75);
[Xtrain, params] = neuramed.features.standardizeFeatures(trainData.X);
Xtest = neuramed.features.standardizeFeatures(testData.X, 'Params', params);

% 4. Train a model (any of 8 algorithms, one consistent interface)
model = neuramed.models.trainRandomForest(Xtrain, trainData.Y, 'NumTrees', 100);

% 5. Evaluate
[pred, probs] = neuramed.models.predictModel(model, Xtest);
report = neuramed.evaluation.generateEvaluationReport(testData.Y, pred, probs);
fprintf('Accuracy: %.3f | AUC: %.3f | Macro-F1: %.3f\n', report.Accuracy, report.AUC, report.MacroF1);

% 6. Clinical decision support
result = neuramed.clinical.predictDisease(model, testData.X(1,:));
rec = neuramed.clinical.generateClinicalRecommendation(result(1));
disp(rec.Summary); disp(rec.SuggestedAction); disp(rec.Disclaimer);

% 7. Persist the model
neuramed.utils.saveModel(model, 'models/rf_v1.mat');
```

See [`examples/`](examples/) for three full end-to-end pipelines (tabular disease classification, signal feature engineering, medical image CNN classification).

## Folder Structure

```
NeuraMedAI-Toolkit/
├── startup.m                      # Adds src/ to MATLAB path; run this first
├── VERSION.txt                    # Current semantic version
├── README.md                      # This file
├── LICENSE
├── RELEASE_NOTES.md
├── src/
│   └── +neuramed/                 # Root package — everything is neuramed.*
│       ├── +dataset/              # Module 1: Dataset Management
│       ├── +features/             # Module 2: Feature Engineering
│       ├── +models/               # Module 3: Machine Learning Models
│       ├── +deeplearning/         # Module 4: Deep Learning Utilities
│       ├── +evaluation/           # Module 5: Model Evaluation
│       ├── +clinical/             # Module 6: Clinical Decision Module
│       └── +utils/                # Module 7: Utilities
│           (each module has a +internal/ sub-package for private helpers)
├── examples/                      # Runnable end-to-end pipelines
├── tests/                         # matlab.unittest test classes, one per module
├── docs/
│   ├── ARCHITECTURE.md
│   └── USER_MANUAL.md
└── .github/workflows/ci.yml       # CI: runs the full test suite on push
```

## Module Reference

| # | Module | Package | Key Functions |
|---|--------|---------|----------------|
| 1 | Dataset Management | `neuramed.dataset` | `loadDataset`, `splitDataset`, `validateData`, `crossValidationSplit`, `augmentData`, `manageMetadata` |
| 2 | Feature Engineering | `neuramed.features` | `normalizeFeatures`, `standardizeFeatures`, `scaleFeatures`, `selectFeatures`, `applyPCA`, `applyLDA`, `reduceDimensionality`, `rankFeatures` |
| 3 | Machine Learning Models | `neuramed.models` | `trainDecisionTree`, `trainRandomForest`, `trainSVM`, `trainKNN`, `trainNaiveBayes`, `trainLogisticRegression`, `trainXGBoost`, `trainEnsemble`, `predictModel` |
| 4 | Deep Learning Utilities | `neuramed.deeplearning` | `buildCNN`, `transferLearningModel`, `createImageDatastore`, `augmentImageData`, `trainingUtilities`, `trainDLModel`, `evaluateDLModel`, `checkGPU` |
| 5 | Model Evaluation | `neuramed.evaluation` | `computeAccuracy`, `computePrecisionRecall`, `computeSpecificitySensitivity`, `computeF1Score`, `computeROCAUC`, `computeConfusionMatrix`, `crossValidateModel`, `generateEvaluationReport` |
| 6 | Clinical Decision Module | `neuramed.clinical` | `predictDisease`, `computeConfidenceScore`, `estimateProbability`, `classifyRisk`, `generateClinicalRecommendation` |
| 7 | Utilities | `neuramed.utils` | `saveModel`, `loadModel`, `manageVersion`, `batchPredict`, `logPerformance`, `manageConfig` |

Full mathematical treatment of every algorithm is in [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md). Design rationale and data-flow contracts are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Design Principles

- **One data contract.** Every dataset — tabular, image, or signal — is passed around as a `struct` with `.X`, `.Y`, `.Type`, `.Source`, `.Metadata`. Every module accepts and returns this contract, so pipelines compose without adapter code.
- **Fit-on-train-only.** Every feature-engineering and dimensionality-reduction function returns a `params` struct from fitting on training data, which is then passed back in to transform test/validation/production data — preventing data leakage by construction, not just by convention.
- **One prediction interface.** `neuramed.models.predictModel(model, X)` works identically regardless of which of the 8 algorithms (or the ensemble) produced `model`. Downstream code (evaluation, clinical decision support, batch prediction) never branches on model type.
- **EPSILON-guarded numerics.** Every division that could hit a zero denominator (variance, IQR, precision/recall denominators) is guarded with a small constant to prevent `NaN`/`Inf` propagation into clinical outputs.
- **Explicit, not silent.** `trainXGBoost` documents that MATLAB has no native XGBoost binding and uses `fitcensemble`'s gradient-boosting methods as the toolbox-native equivalent, rather than silently mislabeling a different algorithm.
- **Clinical outputs are decision support, not diagnosis.** `generateClinicalRecommendation` always attaches a non-diagnostic disclaimer by default; this cannot be silently disabled without explicit opt-out.
- **Reproducibility & auditability.** `manageMetadata` content-hashes datasets, `manageVersion` embeds toolkit version into every saved model, `logPerformance` writes an append-only JSON-lines audit trail — all relevant for eventual regulatory traceability (e.g. IEC 62304 software lifecycle documentation).

## Requirements

- MATLAB R2022b or later (uses `arguments` blocks with validation functions)
- Statistics and Machine Learning Toolbox (all of Module 3, most of Module 5)
- Deep Learning Toolbox (Module 4 only; other modules do not require it)
- Optional: Parallel Computing Toolbox + a CUDA-capable GPU (`neuramed.deeplearning.checkGPU` auto-detects and gracefully falls back to CPU)

## Running Tests

```matlab
cd tests
run_all_tests
```

This uses `matlab.unittest.TestSuite.fromFolder` to discover and run every `test_*_module.m` file and reports a pass/fail summary. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for automated CI on push (requires a self-hosted MATLAB runner, or swap in `matlab-actions/run-tests` with a MathWorks-hosted runner and license).

## Versioning

This toolkit follows [Semantic Versioning](https://semver.org/). Current version: see [`VERSION.txt`](VERSION.txt). See [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for the changelog.

## License

See [`LICENSE`](LICENSE). This toolkit is developed for internal use by Neura-Med AI and its authorized product teams.

## Disclaimer

This toolkit provides AI-assisted decision-support tooling for research and product-development purposes. Outputs from `neuramed.clinical.*` functions are **not** medical diagnoses and must be reviewed by a qualified, licensed clinician before informing any clinical decision. Validate all models on institution-specific data and obtain appropriate regulatory clearance before any clinical deployment.
