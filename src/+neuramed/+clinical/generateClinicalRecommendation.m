function recommendation = generateClinicalRecommendation(predictionResult, options)
%GENERATECLINICALRECOMMENDATION Produce a templated clinical recommendation note.
%
%   REC = neuramed.clinical.generateClinicalRecommendation(PREDICTIONRESULT)
%   takes a result from neuramed.clinical.predictDisease and generates a
%   structured, templated recommendation string. This function produces
%   DECISION-SUPPORT TEXT ONLY: it is explicitly not a diagnosis and must
%   be reviewed by a licensed clinician before any action is taken.
%
%   Name-Value Options:
%       'RiskThresholds' - forwarded to neuramed.clinical.classifyRisk
%       'IncludeDisclaimer' - logical (default true)
%
%   Output REC fields:
%       .Summary        - one-line templated summary
%       .RiskLevel      - categorical tier from classifyRisk
%       .SuggestedAction- generic next-step text keyed to RiskLevel
%       .Disclaimer     - standard non-diagnostic disclaimer text
%
%   Example:
%       result = neuramed.clinical.predictDisease(model, Xpatient);
%       rec = neuramed.clinical.generateClinicalRecommendation(result(1));
%
%   See also NEURAMED.CLINICAL.CLASSIFYRISK, NEURAMED.CLINICAL.PREDICTDISEASE

    arguments
        predictionResult (1,1) struct
        options.RiskThresholds (1,3) double = [0.25 0.5 0.75]
        options.IncludeDisclaimer (1,1) logical = true
    end

    posProb = max(predictionResult.ProbabilityDistribution);
    riskLevel = neuramed.clinical.classifyRisk(posProb, 'Thresholds', options.RiskThresholds);

    actionMap = containers.Map( ...
        {'Low','Moderate','High','Critical'}, ...
        {'Routine follow-up per standard screening interval.', ...
         'Consider additional diagnostic workup and short-interval follow-up.', ...
         'Recommend prompt specialist referral and confirmatory testing.', ...
         'Recommend urgent clinical evaluation without delay.'});

    recommendation = struct();
    recommendation.Summary = sprintf('Predicted: %s (confidence %.1f%%, risk tier: %s)', ...
        predictionResult.DisplayLabel, predictionResult.Confidence * 100, char(riskLevel));
    recommendation.RiskLevel = riskLevel;
    recommendation.SuggestedAction = actionMap(char(riskLevel));

    if options.IncludeDisclaimer
        recommendation.Disclaimer = ['This output is AI-generated decision support and ' ...
            'does NOT constitute a medical diagnosis. All findings must be reviewed and ' ...
            'confirmed by a qualified, licensed clinician before any clinical action is taken.'];
    else
        recommendation.Disclaimer = '';
    end
end
