function model = wrapModel(underlyingModel, modelType, trainOptions)
%WRAPMODEL (internal) Wrap a MATLAB model object in the NeuraMed contract.
%   Standardizes model.Underlying, .Type, .TrainedAt, .Options for use by
%   neuramed.models.predictModel, neuramed.utils.saveModel, and evaluation.
    model = struct();
    model.Underlying = underlyingModel;
    model.Type = modelType;
    model.TrainedAt = datetime('now');
    model.Options = trainOptions;
    model.ToolkitVersion = neuramed.utils.manageVersion('get');
end
