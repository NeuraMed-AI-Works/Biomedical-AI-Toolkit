function out = manageConfig(action, configPathOrStruct, key, value)
%MANAGECONFIG Load, save, or query toolkit/pipeline configuration files.
%
%   CFG = neuramed.utils.manageConfig('load', 'config.json') reads a JSON
%   configuration file into a struct.
%
%   neuramed.utils.manageConfig('save', 'config.json', cfgStruct) writes
%   a struct to a JSON configuration file (pretty-printed).
%
%   VAL = neuramed.utils.manageConfig('get', cfgStruct, 'ModelType') reads
%   a single field, erroring clearly if it is absent (fail-fast on
%   misconfiguration rather than silently using an unintended default).
%
%   Centralizing configuration this way lets pipeline parameters
%   (dataset paths, model hyperparameters, evaluation thresholds) be
%   changed without editing code — an industrial-software best practice
%   for reproducible, auditable biomedical AI pipelines.
%
%   Example:
%       cfg = neuramed.utils.manageConfig('load', 'pipeline_config.json');
%       modelType = neuramed.utils.manageConfig('get', cfg, 'ModelType');

    switch lower(action)
        case 'load'
            configPath = configPathOrStruct;
            if ~exist(configPath, 'file')
                error('neuramed:utils:manageConfig:FileNotFound', ...
                    'Config file "%s" not found.', configPath);
            end
            raw = fileread(configPath);
            out = jsondecode(raw);

        case 'save'
            configPath = configPathOrStruct;
            cfgStruct = key; % third positional arg is the struct to save
            [folder, ~, ~] = fileparts(configPath);
            if ~isempty(folder) && ~isfolder(folder)
                mkdir(folder);
            end
            fid = fopen(configPath, 'w');
            fprintf(fid, '%s', jsonencode(cfgStruct, 'PrettyPrint', true));
            fclose(fid);
            out = configPath;

        case 'get'
            cfgStruct = configPathOrStruct;
            fieldName = key;
            if ~isfield(cfgStruct, fieldName)
                error('neuramed:utils:manageConfig:MissingKey', ...
                    'Configuration key "%s" is not present.', fieldName);
            end
            out = cfgStruct.(fieldName);

        otherwise
            error('neuramed:utils:manageConfig:UnknownAction', ...
                'Unknown action "%s". Use load, save, or get.', action);
    end
end
