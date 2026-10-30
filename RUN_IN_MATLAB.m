%% RUN_IN_MATLAB  One-click verification run for the Neura-Med AI Biomedical AI Toolkit.
%
%   HOW TO USE
%     1. In MATLAB, set the Current Folder to the NeuraMedAI-Toolkit folder
%        (the one containing this file).
%     2. Type:   RUN_IN_MATLAB
%     3. When it finishes, open  run_log.txt  (created in this folder) and
%        send its contents back for analysis.
%
%   What it does: environment/toolbox check -> startup -> full test suite ->
%   example pipelines (tabular + signal). The image CNN example is skipped
%   automatically unless you create examples/sample_images/<Class>/*.jpg.

root = fileparts(mfilename('fullpath'));
cd(root);
logFile = fullfile(root, 'run_log.txt');
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); diary on;
cleanupObj = onCleanup(@() diary('off'));

fprintf('===== ENVIRONMENT =====\n');
fprintf('MATLAB release : %s\n', version('-release'));
fprintf('OS             : %s\n', computer);
fprintf('Date           : %s\n', char(datetime('now')));
required = {'Statistics and Machine Learning Toolbox', 'Deep Learning Toolbox'};
installed = {ver().Name};
for i = 1:numel(required)
    fprintf('%-42s : %s\n', required{i}, ternary(any(strcmp(installed, required{i})), 'INSTALLED', 'MISSING'));
end

fprintf('\n===== STARTUP =====\n');
run(fullfile(root, 'startup.m'));

fprintf('\n===== TEST SUITE =====\n');
try
    cd(fullfile(root, 'tests'));
    run_all_tests;
catch ME
    fprintf(2, 'TEST SUITE ERROR: %s\n%s\n', ME.identifier, ME.message);
    for k = 1:numel(ME.stack)
        fprintf('   at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
    end
end
cd(root);

examples = {'example_disease_classification', 'example_signal_analysis_pipeline', 'example_medical_image_classification'};
for i = 1:numel(examples)
    fprintf('\n===== EXAMPLE: %s =====\n', examples{i});
    try
        cd(fullfile(root, 'examples'));
        run(examples{i});
    catch ME
        fprintf(2, 'EXAMPLE ERROR: %s\n%s\n', ME.identifier, ME.message);
        for k = 1:numel(ME.stack)
            fprintf('   at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
    end
    cd(root);
end

fprintf('\n===== DONE. Log saved to: %s =====\n', logFile);
diary off;

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
