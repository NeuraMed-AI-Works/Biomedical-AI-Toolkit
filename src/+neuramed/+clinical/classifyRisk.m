function riskLevel = classifyRisk(probability, options)
%CLASSIFYRISK Map a disease probability to a discrete clinical risk tier.
%
%   RISKLEVEL = neuramed.clinical.classifyRisk(PROBABILITY) buckets a
%   probability in [0,1] into 'Low' | 'Moderate' | 'High' | 'Critical'
%   using configurable thresholds.
%
%   Name-Value Options:
%       'Thresholds' - 1x3 ascending vector [low_high, moderate_high, high_high]
%                       (default [0.25 0.5 0.75])
%       'Labels'     - 1x4 cell array of tier names
%                       (default {'Low','Moderate','High','Critical'})
%
%   Mathematical Notes:
%       This is a simple threshold classifier on the model's predicted
%       probability p = P(disease | x):
%           Low        if p <  t1
%           Moderate   if t1 <= p < t2
%           High       if t2 <= p < t3
%           Critical   if p >= t3
%       Thresholds should be clinically validated (e.g. via a
%       decision-curve or Youden's-J analysis on a held-out cohort)
%       rather than left at generic defaults for production deployment.
%
%   Example:
%       tier = neuramed.clinical.classifyRisk(0.82);   % -> 'Critical'
%
%   See also NEURAMED.CLINICAL.PREDICTDISEASE

    arguments
        probability
        options.Thresholds (1,3) double = [0.25 0.5 0.75]
        options.Labels cell = {'Low','Moderate','High','Critical'}
    end

    if any(probability < 0 | probability > 1)
        error('neuramed:clinical:classifyRisk:OutOfRange', ...
            'Probability values must lie in [0, 1].');
    end

    t = options.Thresholds;
    L = options.Labels;

    edges = [-Inf, t, Inf];
    idx = discretize(probability, edges);
    riskLevel = categorical(L(idx), L, 'Ordinal', true);
end
