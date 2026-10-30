function model = trainNaiveBayes(X, Y, options)
%TRAINNAIVEBAYES Train a Gaussian Naive Bayes classifier.
%
%   MODEL = neuramed.models.trainNaiveBayes(X, Y) fits a Naive Bayes
%   classifier assuming conditional feature independence given the class.
%
%   Name-Value Options:
%       'DistributionNames' - 'normal' | 'kernel' per-feature distribution
%                              (default 'normal')
%
%   Mathematical Notes:
%       Naive Bayes applies Bayes' rule under the (naive) assumption that
%       features are conditionally independent given the class:
%           P(Y=c | x) proportional to P(Y=c) * prod_j P(x_j | Y=c)
%       For 'normal', P(x_j | Y=c) = N(x_j; mu_{c,j}, sigma_{c,j}^2), with
%       mu and sigma^2 estimated per class per feature from training data.
%       Prediction: yhat = argmax_c [ log P(Y=c) + sum_j log P(x_j|Y=c) ]
%       Despite the independence assumption rarely holding exactly, Naive
%       Bayes is a fast, well-calibrated baseline that performs
%       surprisingly well on high-dimensional biomedical feature sets.
%
%   Example:
%       model = neuramed.models.trainNaiveBayes(Xtrain, Ytrain);
%
%   See also NEURAMED.MODELS.PREDICTMODEL

    arguments
        X
        Y
        options.DistributionNames (1,:) char {mustBeMember(options.DistributionNames, {'normal','kernel'})} = 'normal'
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
    Yc = categorical(Y);

    nb = fitcnb(Xmat, Yc, 'DistributionNames', options.DistributionNames);

    model = neuramed.models.internal.wrapModel(nb, 'NaiveBayes', options);
end
