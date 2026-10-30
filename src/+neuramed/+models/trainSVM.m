function model = trainSVM(X, Y, options)
%TRAINSVM Train a Support Vector Machine classifier.
%
%   MODEL = neuramed.models.trainSVM(X, Y) fits an SVM. For >2 classes,
%   uses one-vs-one via fitcecoc automatically.
%
%   Name-Value Options:
%       'KernelFunction' - 'linear' | 'gaussian' | 'polynomial' | 'rbf' (default 'gaussian')
%       'BoxConstraint'  - regularization C (default 1)
%       'Standardize'    - logical, standardize inputs internally (default true)
%       'PolynomialOrder'- for 'polynomial' kernel (default 3)
%
%   Mathematical Notes:
%       SVM finds the separating hyperplane maximizing the margin
%       between classes, solving:
%           min_{w,b,xi} (1/2)||w||^2 + C * sum_i xi_i
%           s.t. y_i (w'phi(x_i) + b) >= 1 - xi_i,  xi_i >= 0
%       The dual formulation uses a kernel K(x_i, x_j) = phi(x_i)'phi(x_j):
%           max_alpha sum_i alpha_i - (1/2) sum_ij alpha_i alpha_j y_i y_j K(x_i,x_j)
%           s.t. 0 <= alpha_i <= C, sum_i alpha_i y_i = 0
%       Gaussian (RBF) kernel: K(x,z) = exp(-||x-z||^2 / (2*sigma^2)),
%       enabling non-linear decision boundaries. C trades off margin
%       width against training error tolerance (regularization strength).
%
%   Example:
%       model = neuramed.models.trainSVM(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'BoxConstraint', 2);
%
%   See also NEURAMED.MODELS.PREDICTMODEL

    arguments
        X
        Y
        options.KernelFunction (1,:) char {mustBeMember(options.KernelFunction, {'linear','gaussian','rbf','polynomial'})} = 'gaussian'
        options.BoxConstraint (1,1) double {mustBePositive} = 1
        options.Standardize (1,1) logical = true
        options.PolynomialOrder (1,1) double {mustBePositive, mustBeInteger} = 3
    end

    [Xmat, ~, ~] = neuramed.features.internal.toMatrix(X);
    Yc = categorical(Y);
    numClasses = numel(categories(Yc));

    kernel = options.KernelFunction;
    if strcmp(kernel, 'rbf'), kernel = 'gaussian'; end

    svmArgs = {'KernelFunction', kernel, 'BoxConstraint', options.BoxConstraint, ...
        'Standardize', options.Standardize};
    if strcmp(kernel, 'polynomial')
        svmArgs = [svmArgs, {'PolynomialOrder', options.PolynomialOrder}];
    end

    if numClasses > 2
        template = templateSVM(svmArgs{:});
        svmModel = fitcecoc(Xmat, Yc, 'Learners', template);
    else
        svmModel = fitcsvm(Xmat, Yc, svmArgs{:}, 'ClassNames', categories(Yc));
        svmModel = fitPosterior(svmModel);
    end

    model = neuramed.models.internal.wrapModel(svmModel, 'SVM', options);
end
