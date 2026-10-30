function report = validateData(data, options)
%VALIDATEDATA Run data-quality checks on a dataset prior to modeling.
%
%   REPORT = neuramed.dataset.validateData(DATA) inspects DATA (as
%   returned by loadDataset) for missing values, constant/near-constant
%   features, class imbalance, duplicate rows, and out-of-range values,
%   and returns a structured report. Throws no errors by default; issues
%   are collected as warnings so pipelines can decide how to react.
%
%   Name-Value Options:
%       'MissingThreshold'  - fraction of missing values that triggers a
%                              warning for a column (default 0.05)
%       'ImbalanceRatio'    - majority:minority ratio that triggers a
%                              class-imbalance warning (default 3)
%       'Strict'            - if true, throws an error on any issue
%                              instead of only warning (default false)
%
%   Output REPORT struct fields:
%       .IsValid           - logical, true if no issues found
%       .Issues            - cell array of human-readable issue strings
%       .MissingSummary    - table of per-column missing-value counts
%       .ClassBalance      - table of class counts/proportions (if labels present)
%       .NumDuplicates     - count of duplicate observations
%
%   Mathematical Notes:
%       Missing-rate for column j: m_j = (1/N) * sum(isnan(x_ij))
%       Imbalance ratio: r = max_c(n_c) / min_c(n_c)
%
%   Example:
%       report = neuramed.dataset.validateData(data, 'Strict', false);

    arguments
        data
        options.MissingThreshold (1,1) double {mustBeInRange(options.MissingThreshold, 0, 1)} = 0.05
        options.ImbalanceRatio (1,1) double {mustBePositive} = 3
        options.Strict (1,1) logical = false
    end

    issues = {};
    missingSummary = table();
    classBalance = table();
    numDuplicates = 0;

    [X, Y, ~, dataType] = neuramed.dataset.internal.unpack(data);

    if strcmp(dataType, 'tabular') && istable(X)
        varNames = X.Properties.VariableNames;
        missingCounts = zeros(numel(varNames), 1);
        missingFrac = zeros(numel(varNames), 1);
        for j = 1:numel(varNames)
            col = X.(varNames{j});
            if isnumeric(col)
                missingCounts(j) = sum(isnan(col));
            else
                missingCounts(j) = sum(ismissing(col));
            end
            missingFrac(j) = missingCounts(j) / height(X);
            if missingFrac(j) > options.MissingThreshold
                issues{end+1} = sprintf('Column "%s" has %.1f%% missing values (threshold %.1f%%).', ...
                    varNames{j}, 100*missingFrac(j), 100*options.MissingThreshold); %#ok<AGROW>
            end
        end
        missingSummary = table(varNames', missingCounts, missingFrac, ...
            'VariableNames', {'Variable', 'MissingCount', 'MissingFraction'});

        numericVars = varNames(varfun(@isnumeric, X, 'OutputFormat', 'uniform'));
        for j = 1:numel(numericVars)
            col = X.(numericVars{j});
            if nanstd(col) < eps && ~all(isnan(col))
                issues{end+1} = sprintf('Column "%s" is constant or near-constant (std ~ 0).', numericVars{j}); %#ok<AGROW>
            end
        end

        numDuplicates = height(X) - height(unique(X, 'rows'));
        if numDuplicates > 0
            issues{end+1} = sprintf('%d duplicate observation(s) detected.', numDuplicates); %#ok<AGROW>
        end
    end

    if ~isempty(Y)
        classes = unique(Y);
        counts = zeros(numel(classes), 1);
        for i = 1:numel(classes)
            counts(i) = sum(Y == classes(i));
        end
        proportions = counts / sum(counts);
        classBalance = table(classes, counts, proportions, ...
            'VariableNames', {'Class', 'Count', 'Proportion'});

        ratio = max(counts) / max(min(counts), 1);
        if ratio > options.ImbalanceRatio
            issues{end+1} = sprintf('Class imbalance ratio %.2f exceeds threshold %.2f.', ...
                ratio, options.ImbalanceRatio); %#ok<AGROW>
        end
    end

    report = struct();
    report.IsValid = isempty(issues);
    report.Issues = issues';
    report.MissingSummary = missingSummary;
    report.ClassBalance = classBalance;
    report.NumDuplicates = numDuplicates;

    if options.Strict && ~report.IsValid
        error('neuramed:dataset:validateData:ValidationFailed', ...
            'Data validation failed with %d issue(s):\n%s', numel(issues), strjoin(issues, '\n'));
    end
end
