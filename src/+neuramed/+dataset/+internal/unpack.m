function [X, Y, isStructInput, origType] = unpack(data)
%UNPACK (internal) Normalize supported dataset representations to X, Y.
%   Supports: neuramed dataset struct, table, or {X, Y} cell pair.
    if isstruct(data) && isfield(data, 'X')
        X = data.X;
        Y = data.Y;
        isStructInput = true;
        origType = data.Type;
    elseif istable(data)
        X = data;
        Y = [];
        isStructInput = false;
        origType = 'tabular';
    elseif iscell(data) && numel(data) == 2
        X = data{1};
        Y = data{2};
        isStructInput = false;
        origType = 'tabular';
    else
        error('neuramed:dataset:internal:UnsupportedInput', ...
            'Unsupported data representation. Expected a neuramed dataset struct, table, or {X,Y} cell.');
    end
end
