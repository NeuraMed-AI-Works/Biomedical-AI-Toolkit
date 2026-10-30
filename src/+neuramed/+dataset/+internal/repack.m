function out = repack(originalData, X, Y, idx, isStructInput, dataType)
%REPACK (internal) Rebuild a dataset object restricted to indices IDX.
    switch dataType
        case 'image'
            subX = subset(X, idx);
            subY = subX.Labels;
        case 'tabular'
            subX = X(idx, :);
            subY = [];
            if ~isempty(Y)
                subY = Y(idx);
            end
        case 'signal'
            subX = X(idx);
            subY = [];
            if ~isempty(Y)
                subY = Y(idx);
            end
        otherwise
            subX = X(idx, :);
            subY = [];
            if ~isempty(Y)
                subY = Y(idx);
            end
    end

    if isStructInput
        out = originalData;
        out.X = subX;
        out.Y = subY;
    else
        out = struct('X', subX, 'Y', subY, 'Type', dataType, ...
            'Source', 'derived', 'Metadata', struct());
    end
end
