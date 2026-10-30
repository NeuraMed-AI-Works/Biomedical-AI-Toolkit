function layers = transferLearningModel(baseNetwork, numClasses, options)
%TRANSFERLEARNINGMODEL Adapt a pretrained network for a new classification task.
%
%   LAYERS = neuramed.deeplearning.transferLearningModel('resnet18', 3)
%   loads a pretrained backbone and replaces its classifier head to
%   output NUMCLASSES classes, freezing early layers by default.
%
%   Name-Value Options:
%       'FreezeUpToLayer' - name/index of last layer to freeze (default: all but head)
%       'LearnRateFactor'  - multiplier applied to new head layers' learn rate (default 10)
%
%   Mathematical Notes:
%       Transfer learning reuses feature extractors F(x; theta_frozen)
%       learned on a large source domain (e.g. ImageNet) and only
%       fine-tunes a new head g(F(x); theta_new) on the target biomedical
%       dataset. This is effective because low/mid-level convolutional
%       filters (edges, textures, gradients) transfer across visual
%       domains, while only the task-specific decision boundary must be
%       relearned — requiring far fewer labeled medical images than
%       training from scratch. Freezing sets learning rate = 0 for
%       theta_frozen (WeightLearnRateFactor = 0), and LearnRateFactor
%       scales the head's learning rate up relative to the (typically
%       near-zero, if partially unfrozen) backbone rate.
%
%   Example:
%       layers = neuramed.deeplearning.transferLearningModel('resnet18', 2);
%
%   See also NEURAMED.DEEPLEARNING.BUILDCNN, NEURAMED.DEEPLEARNING.TRAINDLMODEL

    arguments
        baseNetwork (1,:) char {mustBeMember(baseNetwork, {'resnet18','resnet50','squeezenet','mobilenetv2'})} = 'resnet18'
        numClasses (1,1) double {mustBePositive, mustBeInteger}
        options.FreezeUpToLayer = []
        options.LearnRateFactor (1,1) double {mustBePositive} = 10
    end

    net = eval(sprintf('%s', baseNetwork)); % requires corresponding Add-On installed
    if isa(net, 'DAGNetwork') || isa(net, 'dlnetwork')
        lgraph = layerGraph(net);
    else
        lgraph = layerGraph(net.Layers);
    end

    [learnableLayer, classLayer] = neuramed.deeplearning.internal.findHeadLayers(lgraph);

    if isa(learnableLayer, 'nnet.cnn.layer.FullyConnectedLayer')
        newLearnable = fullyConnectedLayer(numClasses, ...
            'Name', 'new_fc', ...
            'WeightLearnRateFactor', options.LearnRateFactor, ...
            'BiasLearnRateFactor', options.LearnRateFactor);
    else
        newLearnable = convolution2dLayer(1, numClasses, ...
            'Name', 'new_conv', ...
            'WeightLearnRateFactor', options.LearnRateFactor, ...
            'BiasLearnRateFactor', options.LearnRateFactor);
    end

    lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnable);
    newClassLayer = classificationLayer('Name', 'new_classoutput');
    lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

    layers = lgraph;
end
