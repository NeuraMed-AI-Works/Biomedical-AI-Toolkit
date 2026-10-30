function opts = trainingUtilities(options)
%TRAININGUTILITIES Build a standardized trainingOptions object.
%
%   OPTS = neuramed.deeplearning.trainingUtilities() returns a sensible
%   default trainingOptions('adam', ...) configuration for medical
%   imaging CNNs, auto-detecting GPU availability.
%
%   Name-Value Options:
%       'MaxEpochs'        - default 30
%       'MiniBatchSize'    - default 32
%       'InitialLearnRate' - default 1e-4
%       'ValidationData'   - datastore/table for validation-loss tracking
%       'ValidationFrequency' - default 30
%       'Plots'            - 'training-progress' | 'none' (default 'training-progress')
%
%   Mathematical Notes:
%       Adam optimizer updates parameters using bias-corrected moving
%       averages of the gradient (first moment m_t) and squared gradient
%       (second moment v_t):
%           m_t = beta1*m_{t-1} + (1-beta1)*g_t
%           v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
%           theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
%       giving per-parameter adaptive learning rates that generally
%       converge faster and more robustly than plain SGD for deep
%       CNN training.
%
%   Example:
%       opts = neuramed.deeplearning.trainingUtilities('MaxEpochs', 20);
%       net = trainNetwork(augTrain, layers, opts);

    arguments
        options.MaxEpochs (1,1) double {mustBePositive, mustBeInteger} = 30
        options.MiniBatchSize (1,1) double {mustBePositive, mustBeInteger} = 32
        options.InitialLearnRate (1,1) double {mustBePositive} = 1e-4
        options.ValidationData = []
        options.ValidationFrequency (1,1) double {mustBePositive, mustBeInteger} = 30
        options.Plots (1,:) char {mustBeMember(options.Plots, {'training-progress','none'})} = 'training-progress'
    end

    gpuInfo = neuramed.deeplearning.checkGPU();

    args = {'adam', ...
        'MaxEpochs', options.MaxEpochs, ...
        'MiniBatchSize', options.MiniBatchSize, ...
        'InitialLearnRate', options.InitialLearnRate, ...
        'Shuffle', 'every-epoch', ...
        'ExecutionEnvironment', gpuInfo.Recommendation, ...
        'Plots', options.Plots, ...
        'Verbose', true};

    if ~isempty(options.ValidationData)
        args = [args, {'ValidationData', options.ValidationData, ...
            'ValidationFrequency', options.ValidationFrequency}];
    end

    opts = trainingOptions(args{:});
end
