function info = checkGPU()
%CHECKGPU Report GPU availability and capability for deep learning.
%
%   INFO = neuramed.deeplearning.checkGPU() returns a struct describing
%   whether a CUDA-capable GPU is available to MATLAB and its key
%   properties, so training scripts can select 'ExecutionEnvironment'
%   automatically ('gpu' if available and adequate, else 'cpu').
%
%   Output INFO fields:
%       .Available      - logical
%       .DeviceName     - string, or '' if unavailable
%       .TotalMemoryGB  - double
%       .ComputeCapability - string, e.g. '7.5'
%       .Recommendation - 'gpu' | 'cpu'
%
%   Example:
%       info = neuramed.deeplearning.checkGPU();
%       options = trainingOptions('adam', 'ExecutionEnvironment', info.Recommendation);

    info = struct('Available', false, 'DeviceName', '', ...
        'TotalMemoryGB', 0, 'ComputeCapability', '', 'Recommendation', 'cpu');

    try
        if gpuDeviceCount() > 0
            g = gpuDevice();
            info.Available = true;
            info.DeviceName = g.Name;
            info.TotalMemoryGB = g.TotalMemory / 1e9;
            info.ComputeCapability = g.ComputeCapability;
            info.Recommendation = 'gpu';
        end
    catch
        % Parallel Computing Toolbox not installed or no GPU present.
        info.Available = false;
        info.Recommendation = 'cpu';
    end
end
