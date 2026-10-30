function r = toolkitRoot()
%TOOLKITROOT (internal) Resolve the toolkit root directory from this file's location.
    thisFile = mfilename('fullpath');
    % .../src/+neuramed/+utils/+internal/toolkitRoot.m -> toolkit root is 4 levels up
    d = fileparts(thisFile);
    for i = 1:4
        d = fileparts(d);
    end
    r = d;
end
