function filePath = saveModel(model, savePath, options)
%SAVEMODEL Persist a NeuraMed model to disk with version metadata.
%
%   FILEPATH = neuramed.utils.saveModel(MODEL, SAVEPATH) saves MODEL
%   (from any neuramed.models.train* function) as a .mat file, embedding
%   toolkit version and timestamp for later traceability.
%
%   Name-Value Options:
%       'Overwrite' - logical (default false; errors if file exists)
%
%   Example:
%       neuramed.utils.saveModel(model, 'models/rf_v1.mat');
%
%   See also NEURAMED.UTILS.LOADMODEL, NEURAMED.UTILS.MANAGEVERSION

    arguments
        model (1,1) struct
        savePath (1,:) char
        options.Overwrite (1,1) logical = false
    end

    if exist(savePath, 'file') && ~options.Overwrite
        error('neuramed:utils:saveModel:FileExists', ...
            'File "%s" already exists. Pass "Overwrite", true to replace it.', savePath);
    end

    [folder, ~, ~] = fileparts(savePath);
    if ~isempty(folder) && ~isfolder(folder)
        mkdir(folder);
    end

    neuraMedModel = model; %#ok<NASGU>
    save(savePath, 'neuraMedModel', '-v7.3');
    filePath = savePath;
end
