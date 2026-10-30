function run_all_tests()
%RUN_ALL_TESTS Execute the full Neura-Med AI Biomedical AI Toolkit test suite.
%
%   Run from MATLAB:
%       cd tests
%       run_all_tests
%
%   Exits with an error (non-zero status when run via -batch) if any test
%   fails, so this can be wired into CI (see .github/workflows/ci.yml).

    root = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root, 'startup.m'));

    testFolder = fileparts(mfilename('fullpath'));
    suite = matlab.unittest.TestSuite.fromFolder(testFolder);

    runner = matlab.unittest.TestRunner.withTextOutput('Verbosity', 3);
    results = runner.run(suite);

    fprintf('\n===== Neura-Med AI Toolkit Test Summary =====\n');
    fprintf('Total:  %d\n', numel(results));
    fprintf('Passed: %d\n', nnz([results.Passed]));
    fprintf('Failed: %d\n', nnz([results.Failed]));
    fprintf('Incomplete: %d\n', nnz([results.Incomplete]));

    if any([results.Failed])
        error('neuramed:tests:run_all_tests:Failures', ...
            '%d test(s) failed. See output above for details.', nnz([results.Failed]));
    end
end
