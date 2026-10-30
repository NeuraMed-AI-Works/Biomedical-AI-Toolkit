function h = checksum(data)
%CHECKSUM (internal) Compute a lightweight content hash for a dataset.
    try
        if isstruct(data) && isfield(data, 'X')
            X = data.X;
        else
            X = data;
        end

        if istable(X)
            summaryStr = sprintf('%d_%d_%s', height(X), width(X), strjoin(X.Properties.VariableNames, ','));
        elseif isa(X, 'matlab.io.datastore.ImageDatastore')
            summaryStr = sprintf('imgds_%d', numel(X.Files));
        elseif iscell(X)
            summaryStr = sprintf('cell_%d', numel(X));
        elseif isnumeric(X)
            summaryStr = sprintf('%d_%d_%.6f', size(X,1), size(X,2), sum(X(:), 'omitnan'));
        else
            summaryStr = class(X);
        end

        engine = java.security.MessageDigest.getInstance('SHA-256');
        engine.update(uint8(summaryStr));
        h = sprintf('%02x', typecast(engine.digest(), 'uint8'));
        h = h(1:32); % first 16 bytes as hex for compactness
    catch
        h = 'hash_unavailable';
    end
end
