function model = trainLogisticRegression(X, Y, options)
%TRAINLOGISTICREGRESSION Train a (regularized) logistic regression classifier.
%
%   MODEL = neuramed.models.trainLogisticRegression(X, Y) fits binary or
%   multinomial logistic regression via fitclinear/mnrfit-style GLM.
%
%   Name-Value Options:
%       'Regularization' - 'ridge' | 'lasso' (default 'ridge')
%       'Lambda'         - regularization strength (default 'auto')
%
%   Mathematical Notes:
%       Binary logistic regression models:
%           P(Y=1|x) = sigma(w'x + b) = 1 / (1 + exp(-(w'x+b)))
%       Parameters are fit by maximizing the penalized log-likelihood:
%           max_w sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ] - lambda * R(w)
%       with R(w) = ||w||_2^2 (ridge / L2) encouraging small, diffuse
%       coefficients, or R(w) = ||w||_1 (lasso / L1) encouraging sparse
%       coefficients (automatic feature selection). Multinomial
%       (>2 classes) extends this via the softmax link:
%           P(Y=c|x) = exp(w_c'x) / sum_k exp(w_k'x)
%
%   Example:
%       model = neuramed.models.trainLogisticRegression(Xtrain, Ytrain, 'Regularization', 'lasso');
%
%   See also NEURAMED.MODELS.PREDICTMODEL

    arguments
        X
        Y
        options.Regularization (1,:) char {mustBeMember(options.Regularization, {'ridge','lasso'})} = 'ridge'
        options.Lambda = 'auto'
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
    Yc = categorical(Y);
    numClasses = numel(categories(Yc));

    regName = 'ridge';
    if strcmp(options.Regularization, 'lasso'), regName = 'lasso'; end

    if numClasses > 2
        template = templateLinear('Learner', 'logistic', 'Regularization', regName, 'Lambda', options.Lambda);
        lr = fitcecoc(Xmat, Yc, 'Learners', template, 'Coding', 'onevsall');
    else
        lr = fitclinear(Xmat, Yc, 'Learner', 'logistic', ...
            'Regularization', regName, 'Lambda', options.Lambda);
    end

    model = neuramed.models.internal.wrapModel(lr, 'LogisticRegression', options);
end
