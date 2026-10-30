function model = loadModel(loadPath)
%LOADMODEL Load a NeuraMed model previously saved with saveModel.
%
%   MODEL = neuramed.utils.loadModel(LOADPATH)
%
%   Validates that the file contains a well-formed NeuraMed model
%   (fields: Underlying, Type, TrainedAt) before returning it, and warns
%   if the saved ToolkitVersion differs from the currently installed
%   version (possible breaking changes across versions).
%
%   Example:
%       model = neuramed.utils.loadModel('models/rf_v1.mat');
%
%   See also NEURAMED.UTILS.SAVEMODEL

    if ~exist(loadPath, 'file')
        error('neuramed:utils:loadModel:FileNotFound', ...
            'Model file "%s" not found.', loadPath);
    end

    S = load(loadPath, 'neuraMedModel');
    if ~isfield(S, 'neuraMedModel')
        error('neuramed:utils:loadModel:InvalidFile', ...
            'File "%s" does not contain a variable named "neuraMedModel". Was it saved with neuramed.utils.saveModel?', loadPath);
    end

    model = S.neuraMedModel;
    requiredFields = {'Underlying', 'Type', 'TrainedAt'};
    missing = requiredFields(~isfield(model, requiredFields));
    if ~isempty(missing)
        error('neuramed:utils:loadModel:MalformedModel', ...
            'Loaded model is missing required field(s): %s', strjoin(missing, ', '));
    end

    currentVersion = neuramed.utils.manageVersion('get');
    if isfield(model, 'ToolkitVersion') && ~strcmp(model.ToolkitVersion, currentVersion)
        warning('neuramed:utils:loadModel:VersionMismatch', ...
            'Model was trained with toolkit v%s; current toolkit is v%s. Verify compatibility before production use.', ...
            model.ToolkitVersion, currentVersion);
    end
end
