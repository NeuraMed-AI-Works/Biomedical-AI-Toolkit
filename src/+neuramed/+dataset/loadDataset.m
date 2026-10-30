function data = loadDataset(sourcePath, options)
%LOADDATASET Load a biomedical dataset from disk into a standard struct.
%
%   DATA = neuramed.dataset.loadDataset(SOURCEPATH) loads tabular data
%   (.csv, .xlsx, .mat) or an image folder from SOURCEPATH and returns a
%   standardized dataset struct.
%
%   DATA = neuramed.dataset.loadDataset(SOURCEPATH, OPTIONS) allows
%   name-value options.
%
%   Name-Value Options:
%       'Type'          - 'tabular' | 'image' | 'signal' | 'auto' (default 'auto')
%       'LabelColumn'   - name or index of the label column (tabular only)
%       'IncludeSubfolders' - logical, for image datastores (default true)
%
%   Output DATA struct fields:
%       .X          - feature matrix (N x D) or imageDatastore/signal cell
%       .Y          - label vector (N x 1), categorical when applicable
%       .Type       - 'tabular' | 'image' | 'signal'
%       .Source     - original source path
%       .Metadata   - struct with provenance info (see manageMetadata)
%
%   Mathematical / Computational Notes:
%       No transformation is applied at load time; this function performs
%       pure I/O plus schema normalization so downstream modules can rely
%       on a single consistent data contract.
%
%   Example:
%       data = neuramed.dataset.loadDataset('patients.csv', 'LabelColumn', 'Diagnosis');
%
%   See also NEURAMED.DATASET.SPLITDATASET, NEURAMED.DATASET.VALIDATEDATA

    arguments
        sourcePath (1,:) char
        options.Type (1,:) char {mustBeMember(options.Type, {'tabular','image','signal','auto'})} = 'auto'
        options.LabelColumn = []
        options.IncludeSubfolders (1,1) logical = true
    end

    if ~exist(sourcePath, 'file') && ~exist(sourcePath, 'dir')
        error('neuramed:dataset:loadDataset:PathNotFound', ...
            'Source path "%s" does not exist.', sourcePath);
    end

    resolvedType = options.Type;
    if strcmp(resolvedType, 'auto')
        resolvedType = neuramed.dataset.internal.inferDatasetType(sourcePath);
    end

    data = struct('X', [], 'Y', [], 'Type', resolvedType, ...
        'Source', sourcePath, 'Metadata', struct());

    switch resolvedType
        case 'tabular'
            [~, ~, ext] = fileparts(sourcePath);
            switch lower(ext)
                case '.csv'
                    T = readtable(sourcePath);
                case {'.xlsx', '.xls'}
                    T = readtable(sourcePath);
                case '.mat'
                    S = load(sourcePath);
                    fn = fieldnames(S);
                    if numel(fn) == 1 && istable(S.(fn{1}))
                        T = S.(fn{1});
                    else
                        error('neuramed:dataset:loadDataset:InvalidMat', ...
                            '.mat file must contain exactly one table variable.');
                    end
                otherwise
                    error('neuramed:dataset:loadDataset:UnsupportedFormat', ...
                        'Unsupported tabular file extension "%s".', ext);
            end

            if ~isempty(options.LabelColumn)
                labelCol = options.LabelColumn;
                if isnumeric(labelCol)
                    labelCol = T.Properties.VariableNames{labelCol};
                end
                data.Y = categorical(T.(labelCol));
                T.(labelCol) = [];
            end
            data.X = T;

        case 'image'
            data.X = imageDatastore(sourcePath, ...
                'IncludeSubfolders', options.IncludeSubfolders, ...
                'LabelSource', 'foldernames');
            data.Y = data.X.Labels;

        case 'signal'
            listing = dir(fullfile(sourcePath, '*.mat'));
            if isempty(listing)
                error('neuramed:dataset:loadDataset:NoSignalFiles', ...
                    'No .mat signal files found in "%s".', sourcePath);
            end
            signals = cell(numel(listing), 1);
            for i = 1:numel(listing)
                S = load(fullfile(listing(i).folder, listing(i).name));
                fn = fieldnames(S);
                signals{i} = S.(fn{1});
            end
            data.X = signals;

        otherwise
            error('neuramed:dataset:loadDataset:UnknownType', ...
                'Unable to resolve dataset type for "%s".', sourcePath);
    end

    data.Metadata = neuramed.dataset.manageMetadata('create', data);
end
