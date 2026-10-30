function [learnableLayer, classLayer] = findHeadLayers(lgraph)
%FINDHEADLAYERS (internal) Locate the final learnable and classification layers.
    for i = 1:numel(lgraph.Layers)
        if isa(lgraph.Layers(i), 'nnet.cnn.layer.FullyConnectedLayer') || ...
           isa(lgraph.Layers(i), 'nnet.cnn.layer.Convolution2DLayer')
            learnableLayer = lgraph.Layers(i);
        end
        if isa(lgraph.Layers(i), 'nnet.cnn.layer.ClassificationOutputLayer')
            classLayer = lgraph.Layers(i);
        end
    end
end
