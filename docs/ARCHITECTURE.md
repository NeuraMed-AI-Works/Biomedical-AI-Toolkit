# Architecture — Neura-Med AI Biomedical AI Toolkit

## 1. Design Goals

1. **Reusability across products.** Every Neura-Med AI product (screening tools, wearable inference pipelines, clinical dashboards) should be able to depend on this toolkit without forking it.
2. **One data contract.** A single struct shape flows through every module.
3. **Swappable algorithms.** Any of the 8 classical models, or a CNN, can be substituted in a pipeline without touching evaluation, clinical, or utility code.
4. **No silent data leakage.** Fit/transform separation is enforced by API shape, not by developer discipline alone.
5. **Auditability.** Every trained model and every dataset carries provenance metadata; every training/evaluation run can be logged.

## 2. The Data Contract

All dataset-shaped values in the toolkit are either:

- A **NeuraMed dataset struct**:
  ```matlab
  struct(
      'X', ...,        % table (tabular) | imageDatastore (image) | cell array of signals (signal)
      'Y', ...,        % categorical label vector, or [] if unlabeled
      'Type', ...,     % 'tabular' | 'image' | 'signal'
      'Source', ...,   % string: file path or 'synthetic' / 'derived'
      'Metadata', ...  % struct from neuramed.dataset.manageMetadata
  )
  ```
- A bare **numeric matrix** or **table** (accepted directly by `+features` and `+models` functions for convenience when a caller already has `X`/`Y` in hand and doesn't need the full struct).

`neuramed.dataset.+internal.unpack` / `.repack` are the single choke point that normalizes between these representations, so every function in `+dataset` (split, cross-validate, augment) works identically regardless of which representation was passed in.

## 3. Module Dependency Graph

```
dataset  ─────────────┐
                       │
features ──────────────┼──► models ──► evaluation ──► clinical
                       │       │
deeplearning ──────────┘       └──► utils (saveModel/loadModel/batchPredict wrap any model)
```

- `dataset` has no dependencies on other modules (except calling `utils.manageVersion` for metadata stamping).
- `features` depends only on its own `+internal` helpers (`toMatrix`/`fromMatrix`).
- `models` depends on `features.+internal.toMatrix` to normalize input, and wraps every trained model via `models.+internal.wrapModel`.
- `deeplearning` is largely independent (operates on `imageDatastore`/`layerGraph`/`dlnetwork` objects, not the tabular `X`/`Y` contract), but calls `evaluation.generateEvaluationReport` and `utils.logPerformance` for consistency with the rest of the toolkit.
- `evaluation` depends only on `models.predictModel` (for `crossValidateModel`) and standard label/score arrays — it has no knowledge of *how* a model was trained.
- `clinical` depends on `models.predictModel` and `evaluation`-style probability arrays; it is a thin, clinically-oriented wrapper, not a separate modeling path.
- `utils` is a leaf module that any other module may call (versioning, logging, config, persistence) but which never calls back into them except `models.predictModel` (for `batchPredict`).

This acyclic structure means any module can be unit-tested in isolation with synthetic data, which is exactly how `tests/test_*_module.m` is organized.

## 4. The Model Wrapper Contract

Every `neuramed.models.train*` function returns:

```matlab
struct(
    'Underlying', ...,      % the actual MATLAB model object (ClassificationTree, ClassificationSVM, ...)
    'Type', ...,             % 'DecisionTree' | 'RandomForest' | 'SVM' | 'KNN' | 'NaiveBayes' |
                              % 'LogisticRegression' | 'XGBoost' | 'Ensemble'
    'TrainedAt', ...,        % datetime
    'Options', ...,          % the Name-Value options struct used for training (full reproducibility)
    'ToolkitVersion', ...    % semantic version at training time
)
```

`neuramed.models.predictModel(model, X)` switches on `model.Type`; for every type except `'Ensemble'` this is a direct pass-through to MATLAB's `predict()`. `'Ensemble'` dispatches to `models.+internal.predictEnsemble`, which knows how to combine base-model outputs via soft voting, majority voting, or a stacked meta-classifier.

This means **every** downstream consumer (evaluation, clinical, batch prediction, save/load) only ever needs to call `neuramed.models.predictModel` — none of them contain a switch statement over model type. Adding a 9th algorithm in the future means: (1) write `trainNinthAlgorithm.m` that calls `models.+internal.wrapModel`, (2) add a case to `predictModel.m` only if the underlying object doesn't support MATLAB's standard `predict()` signature (most do, so most new algorithms require *zero* changes to `predictModel.m`).

## 5. Fit-on-Train-Only Pattern

Every function in `+features` that fits statistics from data (`normalizeFeatures`, `standardizeFeatures`, `scaleFeatures`, `applyPCA`, `applyLDA`) follows the same two-mode signature:

```matlab
[Xtransformed, params] = neuramed.features.someTransform(X)                 % FIT + transform
Xtransformed           = neuramed.features.someTransform(X, 'Params', p)    % transform ONLY
```

Pipelines are expected to call the fitting form exactly once, on training data, and the params-only form on every other split (validation, test, production inference). This is enforced by API ergonomics: there is no separate "refit" call that a developer could accidentally invoke on test data without noticing, since supplying `'Params'` always skips fitting.

## 6. Error Handling & Validation Strategy

- **Input validation** uses MATLAB `arguments` blocks with `mustBeMember`, `mustBePositive`, `mustBeInRange`, etc., so invalid calls fail immediately at the API boundary with a clear message, not deep inside a numerical routine.
- **Domain errors** (e.g. non-binary labels passed to `computeROCAUC`, fraction sums that don't add to 1 in `splitDataset`) raise identified errors (`error('neuramed:module:function:Reason', ...)`) so callers can `try/catch` on a specific identifier if needed.
- **Data-quality issues** (missing values, imbalance, duplicates) are reported via `validateData`'s `report.Issues`, not exceptions, by default — because these are frequently expected in real clinical data and the caller should decide whether to proceed. `'Strict', true` escalates them to an error for pipelines that need a hard gate.

## 7. Numerical Safety

Every mathematical function that divides by a data-dependent quantity (`std`, `IQR`, `max - min`, class counts) clamps the denominator to a small `EPSILON` (typically `1e-12`) rather than letting `NaN`/`Inf` propagate silently into a trained model or a clinical probability. This is documented inline in each function's help text under "Mathematical Notes."

## 8. Extensibility Points

- **New model algorithm:** add `neuramed.models.trainX.m` following the `trainDecisionTree.m` template (validate inputs → convert via `features.+internal.toMatrix` → call MATLAB fit function → wrap via `models.+internal.wrapModel`).
- **New feature-engineering method:** add to `+features`, following the fit/params-reuse pattern; register it as a `'Method'` option in `reduceDimensionality.m` if it's a dimensionality-reduction technique, so pipelines can select it via configuration.
- **New evaluation metric:** add to `+evaluation`, taking `(yTrue, yPred[, scores])` and returning either a scalar or a table; wire it into `generateEvaluationReport.m` if it should be part of the standard report.
- **New clinical rule:** extend `classifyRisk.m`'s thresholds/labels via its Name-Value options, or add a new function to `+clinical` for a different rule family (e.g. a composite multi-biomarker risk score) — it should still ultimately consume `models.predictModel`'s output shape.

## 9. Regulatory / Traceability Considerations

While this toolkit does not itself constitute a regulated medical device, it is built with downstream regulatory traceability in mind (e.g. IEC 62304 software-lifecycle documentation, FDA 21 CFR Part 11-style audit trails referenced in prior Neura-Med AI work):

- `manageMetadata` content-hashes every dataset at load time, enabling detection of silent data drift between training and later inference.
- `saveModel`/`loadModel` embed `ToolkitVersion` and `TrainedAt` in every persisted model and warn on load if the toolkit version has changed.
- `logPerformance` writes an append-only, timestamped JSON-lines log of every training/evaluation event, forming an audit trail independent of any single script's console output.
- `generateClinicalRecommendation` cannot silently omit its non-diagnostic disclaimer without an explicit `'IncludeDisclaimer', false` override at the call site.
