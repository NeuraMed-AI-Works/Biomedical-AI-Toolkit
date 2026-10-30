function t = inferDatasetType(sourcePath)
%INFERDATASETTYPE (internal) Heuristically infer dataset type from a path.
%   Not part of the public API. Called by neuramed.dataset.loadDataset.
    if isfolder(sourcePath)
        imgExts = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'};
        hasImages = false;
        for e = imgExts
            if ~isempty(dir(fullfile(sourcePath, '**', ['*' e{1}])))
                hasImages = true;
                break;
            end
        end
        if hasImages
            t = 'image';
        else
            t = 'signal';
        end
    else
        [~, ~, ext] = fileparts(sourcePath);
        switch lower(ext)
            case {'.csv', '.xlsx', '.xls', '.mat'}
                t = 'tabular';
            otherwise
                error('neuramed:dataset:internal:CannotInferType', ...
                    'Cannot infer dataset type from extension "%s". Specify "Type" explicitly.', ext);
        end
    end
end
