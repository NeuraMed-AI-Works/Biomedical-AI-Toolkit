function results = computeROCAUC(yTrue, scores, options)
%COMPUTEROCAUC Compute ROC curve coordinates and Area Under the Curve.
%
%   RESULTS = neuramed.evaluation.computeROCAUC(YTRUE, SCORES) computes
%   the ROC curve for binary classification given continuous scores
%   (posterior probabilities or raw decision scores for the positive
%   class).
%
%   Name-Value Options:
%       'PositiveClass' - label of the positive class (default: second
%                          category alphabetically, or max(yTrue) if numeric)
%
%   Output RESULTS fields:
%       .FPR, .TPR   - vectors of False/True Positive Rate at each threshold
%       .Thresholds  - corresponding decision thresholds
%       .AUC         - scalar Area Under the ROC Curve
%
%   Mathematical Definition:
%       At threshold t, predict positive if score >= t. Sweeping t from
%       max(scores) down to min(scores) traces:
%           TPR(t) = TP(t) / (TP(t)+FN(t)) = Sensitivity(t)
%           FPR(t) = FP(t) / (FP(t)+TN(t)) = 1 - Specificity(t)
%       AUC = integral_0^1 TPR d(FPR), computed here via the trapezoidal
%       rule over the empirical ROC points. AUC has the probabilistic
%       interpretation:
%           AUC = P( score(positive_i) > score(negative_j) )
%       for a randomly drawn positive/negative pair (Mann-Whitney U
%       statistic equivalence), i.e. the probability the classifier
%       ranks a random diseased case above a random healthy case.
%
%   Example:
%       [~, probs] = neuramed.models.predictModel(model, Xtest);
%       results = neuramed.evaluation.computeROCAUC(Ytest, probs(:,2));
%
%   See also NEURAMED.EVALUATION.GENERATEEVALUATIONREPORT

    arguments
        yTrue
        scores (:,1) double
        options.PositiveClass = []
    end

    yTrue = categorical(yTrue);
    classNames = categories(yTrue);
    if numel(classNames) ~= 2
        error('neuramed:evaluation:computeROCAUC:NotBinary', ...
            'ROC/AUC requires exactly 2 classes; found %d. For multiclass, compute one-vs-rest per class.', numel(classNames));
    end

    posClass = options.PositiveClass;
    if isempty(posClass)
        posClass = classNames{end};
    end

    labels = double(yTrue == posClass);
    [fpr, tpr, thresholds, auc] = perfcurve(labels, scores, 1);

    results = struct('FPR', fpr, 'TPR', tpr, 'Thresholds', thresholds, ...
        'AUC', auc, 'PositiveClass', posClass);
end
