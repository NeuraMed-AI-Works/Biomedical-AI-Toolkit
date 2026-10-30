function model = trainXGBoost(X, Y, options)
%TRAINXGBOOST Train a gradient-boosted tree ensemble (XGBoost-equivalent).
%
%   MODEL = neuramed.models.trainXGBoost(X, Y) fits a gradient boosting
%   classifier using MATLAB's fitcensemble with 'LogitBoost'/'GentleBoost',
%   which implements the same functional-gradient-boosting principle as
%   XGBoost (MATLAB has no native XGBoost binding; this is the
%   toolbox-native equivalent and is documented as such rather than
%   silently substituted).
%
%   Name-Value Options:
%       'NumTrees'     - boosting rounds (default 100)
%       'LearnRate'    - shrinkage / eta (default 0.1)
%       'MaxNumSplits' - max splits per weak learner tree (default 20)
%       'Method'       - 'LogitBoost' | 'GentleBoost' | 'AdaBoostM1' (default 'LogitBoost')
%
%   Mathematical Notes:
%       Gradient boosting builds an additive model
%           F_m(x) = F_{m-1}(x) + eta * h_m(x)
%       where each weak learner h_m (a shallow tree) is fit to the
%       negative gradient of the loss w.r.t. the current predictions
%       (functional gradient descent in prediction space):
%           h_m = argmin_h sum_i L(y_i, F_{m-1}(x_i) + h(x_i))
%       For LogitBoost, L is the binomial deviance (log-loss), giving
%       gradient r_i = y_i - p_i(x_i) at each stage — analogous to
%       XGBoost's second-order gradient boosting on the same loss family.
%       LearnRate (eta) shrinks each stage's contribution, trading more
%       boosting rounds for better generalization (regularization via
%       shrinkage).
%
%   Example:
%       model = neuramed.models.trainXGBoost(Xtrain, Ytrain, 'NumTrees', 150, 'LearnRate', 0.05);
%
%   See also NEURAMED.MODELS.TRAINRANDOMFOREST, NEURAMED.MODELS.TRAINENSEMBLE

    arguments
        X
        Y
        options.NumTrees (1,1) double {mustBePositive, mustBeInteger} = 100
        options.LearnRate (1,1) double {mustBePositive} = 0.1
        options.MaxNumSplits (1,1) double {mustBePositive, mustBeInteger} = 20
        options.Method (1,:) char {mustBeMember(options.Method, {'LogitBoost','GentleBoost','AdaBoostM1'})} = 'LogitBoost'
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
    Yc = categorical(Y);

    template = templateTree('MaxNumSplits', options.MaxNumSplits);

    boosted = fitcensemble(Xmat, Yc, ...
        'Method', options.Method, ...
        'NumLearningCycles', options.NumTrees, ...
        'LearnRate', options.LearnRate, ...
        'Learners', template);

    model = neuramed.models.internal.wrapModel(boosted, 'XGBoost', options);
end
