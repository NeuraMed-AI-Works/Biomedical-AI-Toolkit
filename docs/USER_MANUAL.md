# User Manual — Neura-Med AI Biomedical AI Toolkit

Version 1.0.0

This manual documents every function in the toolkit: its purpose, interface, and the mathematics or computation underlying it. For quick API lookup, use MATLAB's `help neuramed.<module>.<function>` — every function's help text is a condensed version of the sections below and is kept in sync with them.

---

## Table of Contents

1. [Module 1 — Dataset Management](#module-1--dataset-management)
2. [Module 2 — Feature Engineering](#module-2--feature-engineering)
3. [Module 3 — Machine Learning Models](#module-3--machine-learning-models)
4. [Module 4 — Deep Learning Utilities](#module-4--deep-learning-utilities)
5. [Module 5 — Model Evaluation](#module-5--model-evaluation)
6. [Module 6 — Clinical Decision Module](#module-6--clinical-decision-module)
7. [Module 7 — Utilities](#module-7--utilities)

---

## Module 1 — Dataset Management

### `loadDataset`
Loads tabular (`.csv`/`.xlsx`/`.mat`), image-folder, or signal-folder (`.mat` per file) data into the standard NeuraMed dataset struct. Performs pure I/O and schema normalization — no statistical transformation — so every downstream function can rely on one input shape. Type is auto-inferred from file extension or folder contents unless `'Type'` is specified explicitly.

### `splitDataset`
Partitions a dataset into train/test(/validation) subsets.

**Stratified sampling.** For each class `c`, indices are shuffled and partitioned so that the proportion of class `c` in every split matches its proportion in the full dataset:
```
n_train_c = floor(p_train * n_c),   n_val_c = floor(p_val * n_c),   remainder -> test
```
This controls for class-imbalance bias that would otherwise make a random split's test accuracy an unreliable estimate of true generalization performance — especially important in biomedical screening, where positive cases are often rare.

### `validateData`
Runs data-quality checks: missing-value rate per column, near-constant columns, duplicate rows, and class-imbalance ratio.

- Missing-rate: `m_j = (1/N) * count(isnan(x_ij))`, flagged if it exceeds `'MissingThreshold'`.
- Imbalance ratio: `r = max_c(n_c) / min_c(n_c)`, flagged if it exceeds `'ImbalanceRatio'`.

Issues are collected into a report rather than thrown by default (`'Strict', false`), since these conditions are common and expected in real clinical data; `'Strict', true` converts them into a hard error for pipelines that require clean input.

### `crossValidationSplit`
Generates K stratified folds. For fold `k`, `Test = F_k`, `Train = union of all other folds`. Averaging a metric across all K folds gives an approximately unbiased estimate of generalization error:
```
CV_error = (1/K) * sum_{k=1}^{K} Error(model_k, F_k)
```
and the standard deviation across folds estimates that error estimate's own variance — critical when working with small biomedical cohorts where a single held-out split can be misleading.

### `augmentData`
For tabular/signal data, applies additive Gaussian jitter scaled to each feature's own standard deviation:
```
x' = x + eps,   eps ~ N(0, (sigma_j * NoiseStd)^2)
```
This variance-preserving perturbation keeps synthetic samples near the true data manifold regardless of each feature's native scale (e.g. age in years vs. a normalized biomarker in [0,1]) — a fixed-scale noise term would over- or under-perturb depending on the feature. For images, applies random rotation/scale/reflection via `imageDataAugmenter`.

### `manageMetadata`
Creates a provenance struct (creation timestamp, type, observation count, toolkit version, and a SHA-256-derived content hash of a data summary) for every loaded/derived dataset. `'verify'` recomputes the hash and compares it to a stored one, detecting silent data drift or tampering between training and later inference — relevant for audit trails in a clinical software context.

---

## Module 2 — Feature Engineering

### `normalizeFeatures` — Min-Max Normalization
```
x' = lo + (x - min(x)) / (max(x) - min(x)) * (hi - lo)
```
Maps each feature column to a fixed range (default `[0,1]`). Columns with zero range are mapped to `lo` (EPSILON-guarded denominator) to avoid division by zero. Sensitive to outliers, since a single extreme value stretches the whole range.

### `standardizeFeatures` — Z-score Standardization
```
x' = (x - mu) / sigma,   mu = mean(x),   sigma = sample std (ddof=1)
```
Produces zero-mean, unit-variance features — the standard preprocessing step for distance-based (KNN), margin-based (SVM), and gradient-based (logistic regression, neural network) algorithms, all of which are scale-sensitive.

### `scaleFeatures` — Robust (Median/IQR) Scaling
```
x' = (x - median(x)) / IQR(x),   IQR(x) = Q3(x) - Q1(x)
```
Robust to outliers because it uses order statistics (median, quartiles) instead of mean/variance, which are themselves sensitive to extreme values. Preferred for clinical lab values that can contain extreme pathological readings.

### `rankFeatures` — Univariate & Multivariate Relevance Ranking
Four selectable methods:

- **`fscore`** (ANOVA F-test): `F_j = Between-class variance / Within-class variance` for feature `j`. High `F_j` means the feature separates classes well relative to its within-class noise.
- **`chi2`**: bins each feature and computes the chi-squared statistic for independence between the binned feature and the class label: `chi2 = sum( (observed - expected)^2 / expected )`.
- **`mrmr`** (Minimum Redundancy Maximum Relevance, via MATLAB's `fscmrmr`): selects features maximizing mutual information with the label while minimizing redundancy with already-selected features:
  ```
  score_j = I(x_j; Y) - (1/|S|) * sum_{s in S} I(x_j; x_s)
  ```
- **`relieff`** (via MATLAB's `relieff`): estimates feature quality from nearest-neighbor margins in feature space; unlike `fscore`, it can capture feature interactions since it looks at local neighborhoods rather than global class statistics.

### `selectFeatures`
Convenience wrapper: ranks via `rankFeatures` and keeps the top-K.

### `applyPCA` — Principal Component Analysis
Given centered data `Xc = X - mean(X)`, PCA diagonalizes the covariance matrix:
```
C = (1/(N-1)) * Xc' * Xc = V * D * V'
```
`V`'s columns are the principal directions (eigenvectors), `D` is diagonal with eigenvalues `lambda_1 >= lambda_2 >= ...` (variance explained by each direction). Projecting onto the top `k` components:
```
X_pca = Xc * V(:, 1:k)
```
`k` is chosen as the smallest value whose cumulative explained variance `r(k) = sum_{i<=k} lambda_i / sum_i lambda_i` exceeds `'VarianceExplained'` (default 0.95), unless `'NumComponents'` is given explicitly. PCA is unsupervised, linear, and variance-preserving — it does not use class labels, so it can discard label-relevant but low-variance directions.

### `applyLDA` — Linear Discriminant Analysis
Unlike PCA, LDA is supervised: it finds projection directions that maximize *class separability* (Fisher's criterion), not just variance. Define:
```
S_B = sum_c N_c * (mu_c - mu)(mu_c - mu)'     (between-class scatter)
S_W = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)'   (within-class scatter)
```
LDA solves the generalized eigenproblem `S_B w = lambda S_W w`; the top `(C-1)` eigenvectors (`C` = number of classes) form the projection matrix. This maximizes `J(w) = (w' S_B w) / (w' S_W w)` — directions along which classes are far apart relative to their internal spread. `S_W` is regularized (`+ EPSILON * I`) for numerical stability when features are collinear or `N` is small relative to `D`.

### `reduceDimensionality`
Unified dispatch over `'pca'` (see above) and `'tsne'`. t-SNE models pairwise similarities as Gaussian conditional probabilities in high-D and Student-t distributed probabilities in low-D, then minimizes their KL divergence via gradient descent:
```
KL(P||Q) = sum_i sum_j p_ij * log(p_ij / q_ij)
```
t-SNE is excellent for 2D/3D visualization of cluster structure but is *transductive* — it has no natural out-of-sample extension, so `'Params'` reuse is not supported for this method (documented explicitly in the function help).

---

## Module 3 — Machine Learning Models

Every function below shares the same call shape: `model = neuramed.models.trainX(X, Y, Name, Value, ...)`, and every trained model is consumed identically via `neuramed.models.predictModel(model, X)`.

### `trainDecisionTree` — CART-style Classification Tree
Greedily selects, at each node, the `(feature, threshold)` split minimizing weighted child impurity. With Gini impurity:
```
Gini(t) = 1 - sum_c p(c|t)^2
Delta = Impurity(parent) - [ (n_L/n) Impurity(L) + (n_R/n) Impurity(R) ]
```
Recursion stops at `'MaxNumSplits'` or `'MinLeafSize'`, which control the bias-variance trade-off — deeper trees fit training data more closely (lower bias, higher variance / overfitting risk).

### `trainRandomForest` — Bagged Tree Ensemble
For `b = 1..B` trees: draw a bootstrap sample (sampling with replacement; each bootstrap holds ~63.2% unique observations), grow a tree considering only a random subset of `m ≈ sqrt(D)` features at each split. Predict by majority vote: `yhat(x) = mode_b{ Tree_b(x) }`. Averaging `B` trees reduces prediction variance by a factor between `1/B` (fully independent trees) and `1` (fully correlated trees); random feature subsampling decorrelates the trees, which is why it matters more than it might first appear. Out-of-bag samples (~36.8% not drawn per bootstrap) give a built-in unbiased error estimate without needing a separate held-out set.

### `trainSVM` — Support Vector Machine
Finds the maximum-margin separating hyperplane:
```
min_{w,b,xi} (1/2)||w||^2 + C * sum_i xi_i
s.t. y_i (w'phi(x_i)+b) >= 1 - xi_i,  xi_i >= 0
```
solved via its dual with kernel `K(x_i,x_j) = phi(x_i)'phi(x_j)`. The Gaussian/RBF kernel `K(x,z) = exp(-||x-z||^2 / (2 sigma^2))` enables non-linear boundaries. `'BoxConstraint'` (`C`) trades off margin width against training-error tolerance. For >2 classes, MATLAB's `fitcecoc` combines multiple binary SVMs via error-correcting output codes (effectively one-vs-one).

### `trainKNN` — K-Nearest Neighbors
Non-parametric: classifies a query `x` by (weighted) majority vote among its `K` nearest training points under a chosen distance metric:
```
yhat = argmax_c sum_{i in N_K(x)} w_i * 1[y_i = c]
```
with `w_i = 1` (equal), `1/d_i` (inverse), or `1/d_i^2` (squaredinverse). Decision boundaries adapt directly to local data density at the cost of `O(N)` prediction time and sensitivity to feature scale — hence `'Standardize', true` by default.

### `trainNaiveBayes` — Gaussian Naive Bayes
Applies Bayes' rule under conditional feature independence given the class:
```
P(Y=c|x) proportional to P(Y=c) * prod_j P(x_j|Y=c)
```
With `'normal'` distributions, `P(x_j|Y=c) = N(x_j; mu_{c,j}, sigma_{c,j}^2)` fit per class per feature. Prediction: `yhat = argmax_c [ log P(Y=c) + sum_j log P(x_j|Y=c) ]`. A fast, well-calibrated baseline that often performs surprisingly well on high-dimensional biomedical feature sets despite the independence assumption rarely holding exactly.

### `trainLogisticRegression`
Binary case models `P(Y=1|x) = sigma(w'x+b) = 1/(1+exp(-(w'x+b)))`, fit by maximizing penalized log-likelihood:
```
max_w sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)] - lambda * R(w)
```
`R(w) = ||w||_2^2` (ridge/L2, default) shrinks coefficients smoothly; `R(w) = ||w||_1` (lasso/L1) produces sparse coefficients (automatic feature selection). Multinomial (>2 classes) extends this via the softmax link `P(Y=c|x) = exp(w_c'x) / sum_k exp(w_k'x)`.

### `trainXGBoost` — Gradient-Boosted Trees
MATLAB has no native XGBoost binding; this function uses `fitcensemble` with `'LogitBoost'`/`'GentleBoost'`/`'AdaBoostM1'`, MATLAB's toolbox-native gradient-boosting implementations, and documents this substitution explicitly rather than silently mislabeling the algorithm. Gradient boosting builds an additive model:
```
F_m(x) = F_{m-1}(x) + eta * h_m(x)
```
where each weak learner `h_m` is fit to the negative gradient of the loss with respect to current predictions (functional gradient descent). For LogitBoost this is binomial deviance, giving gradient `r_i = y_i - p_i(x_i)` — the same loss family XGBoost optimizes with second-order (Newton) boosting. `'LearnRate'` (`eta`) shrinks each stage's contribution, trading more boosting rounds for better generalization.

### `trainEnsemble` — Multi-Model Voting / Stacking
Trains several base learners and combines them:

- **`softvote`**: average class posterior probabilities across `M` models: `P_ens(c|x) = (1/M) sum_m P_m(c|x)`, predict `argmax_c`. Reduces variance when base models make partially uncorrelated errors — under pairwise error correlation `rho`, variance reduction ranges from a factor of `1/M` (`rho=0`) to `1` (`rho=1`).
- **`majorityvote`**: hard-label mode across base models.
- **`stacking`**: trains a logistic-regression meta-classifier on the base models' predicted probabilities as new features, letting the ensemble learn how to weight base models rather than averaging them uniformly.

### `predictModel`
Single dispatch point for prediction across all 8 model types plus ensembles. Every other module (evaluation, clinical, batch prediction) calls only this function — never a model-type-specific predict call — which is what allows the toolkit to add new algorithms without touching downstream code (see `docs/ARCHITECTURE.md` §4).

---

## Module 4 — Deep Learning Utilities

### `buildCNN`
Constructs a configurable convolution → batch-norm → ReLU → max-pool block stack, ending in dropout → fully-connected → softmax → classification output.

- **Convolution**: `y[i,j,k] = sum_{m,n,c} W[m,n,c,k] * x[i+m,j+n,c] + b_k` — learns local, translation-shared feature detectors.
- **ReLU**: `f(z) = max(0,z)` — non-linearity enabling hierarchical feature composition.
- **Batch normalization**: `z' = (z - batch_mean)/sqrt(batch_var + eps)` — stabilizes and accelerates training by controlling internal covariate shift.
- **Max-pooling**: downsamples via local-window maximum, providing translation invariance and reducing computation.
- **Dropout**: zeroes activations with probability `p` during training; approximates averaging over `2^n` thinned sub-networks (a stochastic regularizer, with a Bayesian-approximation interpretation), reducing overfitting — especially important for typically small labeled biomedical imaging datasets.

### `transferLearningModel`
Loads a pretrained backbone (`resnet18`, `resnet50`, `squeezenet`, `mobilenetv2`) and replaces its classification head. Low/mid-level convolutional filters (edges, textures, gradients) transfer across visual domains; only the task-specific decision boundary needs relearning, drastically reducing the labeled-data requirement relative to training from scratch — the standard approach for medical imaging tasks where large labeled cohorts are expensive to obtain. `'LearnRateFactor'` scales the new head's learning rate relative to the (frozen or near-frozen) backbone.

### `createImageDatastore` / `augmentImageData`
Build a labeled `imageDatastore` from a class-subfoldered directory, and wrap it in on-the-fly augmentation (`imageDataAugmenter` + `augmentedImageDatastore`). Each image gets a freshly sampled random affine transform `T = Translate(t) * Rotate(theta) * Scale(s)` at *every* epoch, not a fixed precomputed augmented set — this effectively trains on a continuous augmentation manifold around each original sample, reducing overfitting risk for small biomedical imaging cohorts.

### `checkGPU` / `trainingUtilities`
`checkGPU` detects CUDA-capable GPU availability and reports device properties, feeding a `'gpu'`/`'cpu'` recommendation into `trainingUtilities`, which builds a standardized `trainingOptions('adam', ...)` object. The Adam optimizer uses bias-corrected moving averages of the gradient and squared gradient:
```
m_t = beta1*m_{t-1} + (1-beta1)*g_t
v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```
giving per-parameter adaptive learning rates that typically converge faster and more robustly than plain SGD for deep CNN training.

### `trainDLModel` / `evaluateDLModel`
Thin wrappers around `trainNetwork`/`classify` that add standardized performance logging (`utils.logPerformance`) and route evaluation through the same `evaluation.generateEvaluationReport` used by classical ML models, so deep-learning and classical pipelines produce directly comparable reports.

---

## Module 5 — Model Evaluation

### `computeConfusionMatrix`
`C(i,j)` = count of true-class-`i` observations predicted as class `j`. For binary problems with positive class as row/column 2: `TN=C(1,1), FP=C(1,2), FN=C(2,1), TP=C(2,2)`.

### `computeAccuracy`
`Accuracy = (TP+TN) / (TP+TN+FP+FN)`. Can be misleading under class imbalance (e.g. disease screening, where the negative class dominates) — a model predicting "no disease" for everyone can still score high accuracy while missing every true case. Prefer Sensitivity/Specificity/F1/AUC for imbalanced clinical data.

### `computePrecisionRecall`
Per class `c`: `Precision_c = TP_c/(TP_c+FP_c)` ("of predicted `c`, how many were right"), `Recall_c = TP_c/(TP_c+FN_c)` ("of actual `c`, how many were found"). Macro-average treats every class equally (robust to imbalance in the *reporting*, though not in training); micro-average pools all TP/FP/FN globally, so it is dominated by the majority class.

### `computeSpecificitySensitivity`
The standard clinical-diagnostic metric pair. `Sensitivity_c = TP_c/(TP_c+FN_c)` = "probability the test is positive given disease present" = Recall. `Specificity_c = TN_c/(TN_c+FP_c)` = "probability the test is negative given disease absent." Separately quantifying miss rate (`1-Sensitivity`) and false-alarm rate (`1-Specificity`) is why these, not accuracy, are the primary reported metrics in clinical-diagnostic literature.

### `computeF1Score`
`F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)` — the harmonic mean, which penalizes models that trade one off heavily against the other (unlike the arithmetic mean, which would not). Macro-F1 is a standard single-number summary for imbalanced multi-class problems.

### `computeROCAUC`
Sweeping the decision threshold `t` from `max(scores)` to `min(scores)` traces:
```
TPR(t) = Sensitivity(t),   FPR(t) = 1 - Specificity(t)
```
`AUC = integral_0^1 TPR d(FPR)` (computed via the trapezoidal rule over empirical ROC points using `perfcurve`). AUC has a clean probabilistic interpretation — it equals the probability that the classifier ranks a randomly drawn positive case above a randomly drawn negative case (the Mann-Whitney U-statistic equivalence), which is why it is threshold-independent and widely reported for diagnostic-test performance.

### `crossValidateModel`
Runs `neuramed.dataset.crossValidationSplit`, trains a given `trainFcn` on each fold's training data, evaluates on the held-out fold, and aggregates mean/std of Accuracy and Macro-F1 across folds — the standard deviation serving as an empirical estimate of generalization-error variance, particularly informative for small biomedical cohorts.

### `generateEvaluationReport`
One-call aggregation of all of the above into a single struct, used identically by classical ML (Module 3) and deep learning (Module 4) pipelines for a consistent reporting format across the entire toolkit.

---

## Module 6 — Clinical Decision Module

### `predictDisease`
Runs `models.predictModel`, then packages the result into a clinically interpretable struct array (`PredictedClass`, `DisplayLabel`, `Confidence`, `ProbabilityDistribution`) — the entry point clinical-facing UIs should call rather than working with raw model output directly.

### `computeConfidenceScore`
`confidence_i = max_c P(Y=c|x_i)` — the probability mass on the predicted class. The simplest, most interpretable calibrated-confidence proxy.

### `estimateProbability`
Returns the model's posterior probability matrix plus per-observation Shannon entropy:
```
H(x) = - sum_c P(c|x) * log2(P(c|x))
```
`H=0` under full certainty (one class has probability 1); `H` is maximal (`log2(C)`) under a uniform distribution over `C` classes. Useful for flagging cases that may warrant clinician review rather than fully automated triage, independent of which specific class was predicted.

### `classifyRisk`
Threshold classifier mapping a probability `p = P(disease|x)` to a discrete tier:
```
Low if p<t1,  Moderate if t1<=p<t2,  High if t2<=p<t3,  Critical if p>=t3
```
Default thresholds `[0.25, 0.5, 0.75]` are generic placeholders. **Thresholds must be clinically validated** (e.g. via decision-curve analysis or Youden's-J statistic on a held-out cohort) before any production/clinical deployment — this is documented explicitly in the function's help text and is a required calibration step, not an optional tuning choice.

### `generateClinicalRecommendation`
Produces a templated recommendation (`Summary`, `RiskLevel`, `SuggestedAction`, `Disclaimer`) from a `predictDisease` result. **This function generates decision-support text only — never a diagnosis.** The disclaimer (*"does NOT constitute a medical diagnosis... must be reviewed and confirmed by a qualified, licensed clinician"*) is attached by default and requires an explicit `'IncludeDisclaimer', false` override to omit, which should never be done in a clinician-facing deployment.

---

## Module 7 — Utilities

### `saveModel` / `loadModel`
Persist/restore the full model wrapper struct (`.mat`, `-v7.3`), embedding `ToolkitVersion` and `TrainedAt`. `loadModel` validates the loaded struct's shape and warns if the saved toolkit version differs from the currently installed one, since behavior of underlying MATLAB fit functions can change across MATLAB/toolbox releases.

### `manageVersion`
Get/set the toolkit's semantic version (`MAJOR.MINOR.PATCH[-prerelease]`), persisted in `VERSION.txt` at the toolkit root. MAJOR = incompatible API changes, MINOR = backward-compatible additions, PATCH = backward-compatible fixes.

### `batchPredict`
Chunks a large `X` into `'BatchSize'`-row batches and calls `predictModel` per batch, concatenating results — avoids materializing an entire large clinical registry's predictions in one pass and reports progress, useful for hundreds-of-thousands-of-record batch scoring jobs.

### `logPerformance`
Appends a timestamped JSON object (event name, toolkit version, arbitrary metrics struct) to an append-only `.jsonl` log file — an audit trail independent of any single script's console output, useful for regulatory traceability of training/evaluation runs over a product's lifecycle.

### `manageConfig`
Load/save/query JSON configuration files, so pipeline parameters (dataset paths, model hyperparameters, evaluation thresholds) can be changed without editing code. `'get'` fails fast (clear error) on a missing key rather than silently falling back to an unintended default — deliberately avoiding a common source of silent misconfiguration in production pipelines.
