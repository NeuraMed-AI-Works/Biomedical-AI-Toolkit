function result = manageMetadata(action, target, varargin)
%MANAGEMETADATA Create, update, or query dataset provenance metadata.
%
%   META = neuramed.dataset.manageMetadata('create', DATA) builds a
%   metadata struct capturing shape, type, timestamp, and a content hash
%   for reproducibility and audit trails (important for clinical/
%   regulatory traceability).
%
%   META = neuramed.dataset.manageMetadata('update', META, 'Key', VALUE, ...)
%   adds/overwrites custom fields (e.g. 'Source', 'Version', 'Notes').
%
%   TF = neuramed.dataset.manageMetadata('verify', DATA, META) recomputes
%   the content hash of DATA and checks it against META.ContentHash,
%   returning true if the data is unchanged since metadata creation.
%
%   Mathematical Notes:
%       ContentHash uses a SHA-256 digest of a serialized summary
%       (size, type, and a numeric checksum of sampled values) so that
%       tampering or silent data drift can be detected without storing
%       the full dataset twice.
%
%   Example:
%       meta = neuramed.dataset.manageMetadata('create', data);
%       meta = neuramed.dataset.manageMetadata('update', meta, 'Version', '1.0.0');

    switch lower(action)
        case 'create'
            data = target;
            meta = struct();
            meta.CreatedAt = datetime('now');
            meta.Type = getfield_safe(data, 'Type', 'unknown');
            meta.NumObservations = neuramed.dataset.internal.numObservations( ...
                getfield_safe(data, 'X', []), meta.Type);
            meta.ContentHash = neuramed.dataset.internal.checksum(data);
            meta.ToolkitVersion = neuramed.utils.manageVersion('get');
            meta.CustomFields = struct();
            result = meta;

        case 'update'
            meta = target;
            if mod(numel(varargin), 2) ~= 0
                error('neuramed:dataset:manageMetadata:BadArgs', ...
                    'Update requires Key,Value pairs.');
            end
            for i = 1:2:numel(varargin)
                meta.CustomFields.(varargin{i}) = varargin{i+1};
            end
            result = meta;

        case 'verify'
            data = target;
            meta = varargin{1};
            currentHash = neuramed.dataset.internal.checksum(data);
            result = strcmp(currentHash, meta.ContentHash);

        otherwise
            error('neuramed:dataset:manageMetadata:UnknownAction', ...
                'Unknown action "%s". Use create, update, or verify.', action);
    end
end

function v = getfield_safe(s, f, default)
    if isstruct(s) && isfield(s, f)
        v = s.(f);
    else
        v = default;
    end
end
