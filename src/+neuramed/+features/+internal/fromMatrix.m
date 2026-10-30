function Xout = fromMatrix(Xmat, isTable, varNames, original)
%FROMMATRIX (internal) Convert a numeric matrix back to original container type.
    if isTable
        Xout = array2table(Xmat, 'VariableNames', varNames);
    else
        Xout = Xmat;
    end
end
