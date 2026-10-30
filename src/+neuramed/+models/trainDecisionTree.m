function model = trainDecisionTree(X, Y, options)
%TRAINDECISIONTREE Train a classification decision tree (CART-style).
%
%   MODEL = neuramed.models.trainDecisionTree(X, Y) fits a decision tree
%   classifier and returns a NeuraMed model wrapper.
%
%   Name-Value Options:
%       'MaxNumSplits'  - maximum tree splits (default 20, controls overfit)
%       'MinLeafSize'   - minimum observations per leaf (default 5)
%       'SplitCriterion'- 'gdi' (Gini) | 'deviance' | 'twoing' (default 'gdi')
%
%   Mathematical Notes:
%       At each node, the algorithm greedily selects the (feature,
%       threshold) pair minimizing the weighted child impurity:
%           Gini(t) = 1 - sum_c p(c|t)^2
%       Split gain: Delta = Impurity(parent) - [ (n_L/n) Impurity(L) + (n_R/n) Impurity(R) ]
%       Recursion stops at MaxNumSplits or MinLeafSize, controlling the
%       bias-variance trade-off of the tree.
%
%   Example:
%       model = neuramed.models.trainDecisionTree(Xtrain, Ytrain, 'MaxNumSplits', 15);
%       pred = neuramed.models.predictModel(model, Xtest);
%
%   See also NEURAMED.MODELS.TRAINRANDOMFOREST, NEURAMED.MODELS.PREDICTMODEL

    arguments
        X
        Y
        options.MaxNumSplits (1,1) double {mustBePositive, mustBeInteger} = 20
        options.MinLeafSize (1,1) double {mustBePositive, mustBeInteger} = 5
        options.SplitCriterion (1,:) char {mustBeMember(options.SplitCriterion, {'gdi','deviance','twoing'})} = 'gdi'
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
    Yc = categorical(Y);

    tree = fitctree(Xmat, Yc, ...
        'MaxNumSplits', options.MaxNumSplits, ...
        'MinLeafSize', options.MinLeafSize, ...
        'SplitCriterion', options.SplitCriterion);

    model = neuramed.models.internal.wrapModel(tree, 'DecisionTree', options);
end
