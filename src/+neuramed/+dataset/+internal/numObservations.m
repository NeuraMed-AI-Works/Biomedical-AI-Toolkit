function n = numObservations(X, dataType)
%NUMOBSERVATIONS (internal) Count observations for supported data types.
    switch dataType
        case 'image'
            n = numel(X.Files);
        case 'tabular'
            n = height(X);
        case 'signal'
            n = numel(X);
        otherwise
            n = size(X, 1);
    end
end
