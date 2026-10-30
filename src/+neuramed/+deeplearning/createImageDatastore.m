function imds = createImageDatastore(folderPath, options)
%CREATEIMAGEDATASTORE Build a labeled imageDatastore from a folder tree.
%
%   IMDS = neuramed.deeplearning.createImageDatastore(FOLDERPATH) expects
%   subfolders named by class (e.g. FOLDERPATH/Benign, FOLDERPATH/Malignant)
%   and returns a labeled imageDatastore.
%
%   Name-Value Options:
%       'IncludeSubfolders' - logical (default true)
%       'FileExtensions'    - cell array (default {'.jpg','.png','.tif','.jpeg'})
%       'ReadSize'          - batch read size (default 1)
%
%   Example:
%       imds = neuramed.deeplearning.createImageDatastore('data/xrays');
%
%   See also NEURAMED.DATASET.LOADDATASET, NEURAMED.DEEPLEARNING.AUGMENTIMAGEDATA

    arguments
        folderPath (1,:) char
        options.IncludeSubfolders (1,1) logical = true
        options.FileExtensions cell = {'.jpg','.png','.tif','.jpeg','.bmp'}
        options.ReadSize (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    if ~isfolder(folderPath)
        error('neuramed:deeplearning:createImageDatastore:FolderNotFound', ...
            'Folder "%s" does not exist.', folderPath);
    end

    imds = imageDatastore(folderPath, ...
        'IncludeSubfolders', options.IncludeSubfolders, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', options.FileExtensions, ...
        'ReadSize', options.ReadSize);

    if isempty(imds.Files)
        error('neuramed:deeplearning:createImageDatastore:Empty', ...
            'No images found under "%s" with extensions %s.', ...
            folderPath, strjoin(options.FileExtensions, ', '));
    end
end
