function [Xmat, isTable, varNames] = toMatrix(X)
%TOMATRIX (internal) Convert table or numeric array to a numeric matrix.
    if istable(X)
        isTable = true;
        varNames = X.Properties.VariableNames;
        isNumericVar = varfun(@isnumeric, X, 'OutputFormat', 'uniform');
        if ~all(isNumericVar)
            error('neuramed:features:internal:NonNumericColumns', ...
                'Table contains non-numeric columns: %s', ...
                strjoin(varNames(~isNumericVar), ', '));
        end
        Xmat = X{:, :};
    elseif isnumeric(X)
        isTable = false;
        varNames = {};
        Xmat = X;
    else
        error('neuramed:features:internal:UnsupportedInput', ...
            'Input must be a numeric matrix or a numeric table.');
    end
end
