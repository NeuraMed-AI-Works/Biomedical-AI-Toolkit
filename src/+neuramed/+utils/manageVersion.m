function out = manageVersion(action, newVersion)
%MANAGEVERSION Get or set the Neura-Med AI Biomedical AI Toolkit version.
%
%   V = neuramed.utils.manageVersion('get') returns the current toolkit
%   semantic version string, read from VERSION.txt at the toolkit root.
%
%   neuramed.utils.manageVersion('set', '1.1.0') updates VERSION.txt
%   (intended for maintainers cutting a new release, not runtime use).
%
%   Semantic Versioning (MAJOR.MINOR.PATCH):
%       MAJOR - incompatible API changes
%       MINOR - backward-compatible functionality additions
%       PATCH - backward-compatible bug fixes
%
%   Example:
%       v = neuramed.utils.manageVersion('get');

    versionFile = fullfile(neuramed.utils.internal.toolkitRoot(), 'VERSION.txt');

    switch lower(action)
        case 'get'
            if exist(versionFile, 'file')
                out = strtrim(fileread(versionFile));
            else
                out = '0.0.0-unknown';
            end
        case 'set'
            if nargin < 2
                error('neuramed:utils:manageVersion:MissingVersion', ...
                    'Provide a version string, e.g. manageVersion(''set'', ''1.1.0'').');
            end
            if isempty(regexp(newVersion, '^\d+\.\d+\.\d+(-[\w.]+)?$', 'once'))
                error('neuramed:utils:manageVersion:InvalidFormat', ...
                    'Version "%s" is not valid semantic versioning (MAJOR.MINOR.PATCH).', newVersion);
            end
            fid = fopen(versionFile, 'w');
            fprintf(fid, '%s', newVersion);
            fclose(fid);
            out = newVersion;
        otherwise
            error('neuramed:utils:manageVersion:UnknownAction', ...
                'Unknown action "%s". Use "get" or "set".', action);
    end
end
