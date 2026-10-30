function logPerformance(eventName, metrics, options)
%LOGPERFORMANCE Append a structured performance log entry (JSON Lines).
%
%   neuramed.utils.logPerformance('ModelTraining', metricsStruct) appends
%   a timestamped JSON record to the toolkit's performance log, giving an
%   auditable history of training/evaluation runs — useful for
%   regulatory traceability in clinical software.
%
%   Name-Value Options:
%       'LogFile' - path to log file (default: <toolkitRoot>/logs/performance.jsonl)
%
%   Example:
%       neuramed.utils.logPerformance('CrossValidation', ...
%           struct('MeanAccuracy', 0.94, 'MeanF1', 0.91, 'Model', 'RandomForest'));
%
%   See also NEURAMED.UTILS.MANAGECONFIG

    arguments
        eventName (1,:) char
        metrics (1,1) struct
        options.LogFile (1,:) char = ''
    end

    if isempty(options.LogFile)
        logDir = fullfile(neuramed.utils.internal.toolkitRoot(), 'logs');
        if ~isfolder(logDir)
            mkdir(logDir);
        end
        options.LogFile = fullfile(logDir, 'performance.jsonl');
    end

    entry = struct();
    entry.Timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
    entry.Event = eventName;
    entry.ToolkitVersion = neuramed.utils.manageVersion('get');
    entry.Metrics = metrics;

    jsonLine = jsonencode(entry);

    fid = fopen(options.LogFile, 'a');
    if fid == -1
        warning('neuramed:utils:logPerformance:CannotOpenLog', ...
            'Could not open log file "%s"; performance event not logged.', options.LogFile);
        return
    end
    fprintf(fid, '%s\n', jsonLine);
    fclose(fid);
end
