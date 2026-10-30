function startup()
%STARTUP Initialize the Neura-Med AI Biomedical AI Toolkit on the MATLAB path.
%
%   Run this once per session (or add to your MATLAB startup.m) before
%   using any neuramed.* functions:
%
%       run('path/to/NeuraMedAI-Toolkit/startup.m')
%
%   This adds the toolkit's src/ directory to the path so the
%   neuramed.dataset, neuramed.features, neuramed.models,
%   neuramed.deeplearning, neuramed.evaluation, neuramed.clinical, and
%   neuramed.utils packages are available.

    root = fileparts(mfilename('fullpath'));
    srcPath = fullfile(root, 'src');

    if ~isfolder(srcPath)
        error('neuramed:startup:SrcNotFound', ...
            'Could not find src/ directory at "%s".', srcPath);
    end

    addpath(srcPath);

    v = neuramed.utils.manageVersion('get');
    fprintf('Neura-Med AI Biomedical AI Toolkit v%s initialized.\n', v);
    fprintf('  Package root added to path: %s\n', srcPath);
    fprintf('  Modules: dataset | features | models | deeplearning | evaluation | clinical | utils\n');
end
