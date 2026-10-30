function model = trainKNN(X, Y, options)
%TRAINKNN Train a K-Nearest-Neighbors classifier.
%
%   MODEL = neuramed.models.trainKNN(X, Y) fits a KNN classifier.
%
%   Name-Value Options:
%       'NumNeighbors' - K (default 5)
%       'Distance'     - 'euclidean' | 'cityblock' | 'cosine' | 'mahalanobis' (default 'euclidean')
%       'DistanceWeight' - 'equal' | 'inverse' | 'squaredinverse' (default 'equal')
%       'Standardize'  - logical (default true)
%
%   Mathematical Notes:
%       For a query x, find the K training points minimizing distance
%       d(x, x_i), e.g. Euclidean d(x,x_i) = sqrt(sum_j (x_j - x_ij)^2).
%       Classify by (weighted) majority vote:
%           yhat = argmax_c sum_{i in N_K(x)} w_i * 1[y_i = c]
%       with w_i = 1 (equal), 1/d_i (inverse), or 1/d_i^2 (squaredinverse).
%       KNN is non-parametric: decision boundaries adapt directly to
%       local data density, at the cost of O(N) prediction complexity
%       and sensitivity to feature scale (hence Standardize default true).
%
%   Example:
%       model = neuramed.models.trainKNN(Xtrain, Ytrain, 'NumNeighbors', 7);
%
%   See also NEURAMED.MODELS.PREDICTMODEL

    arguments
        X
        Y
        options.NumNeighbors (1,1) double {mustBePositive, mustBeInteger} = 5
        options.Distance (1,:) char = 'euclidean'
        options.DistanceWeight (1,:) char {mustBeMember(options.DistanceWeight, {'equal','inverse','squaredinverse'})} = 'equal'
        options.Standardize (1,1) logical = true
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
    Yc = categorical(Y);

    knn = fitcknn(Xmat, Yc, ...
        'NumNeighbors', options.NumNeighbors, ...
        'Distance', options.Distance, ...
        'DistanceWeight', options.DistanceWeight, ...
        'Standardize', options.Standardize);

    model = neuramed.models.internal.wrapModel(knn, 'KNN', options);
end
