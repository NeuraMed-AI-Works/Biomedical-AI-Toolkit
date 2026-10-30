# How to Run the Neura-Med AI Biomedical AI Toolkit in MATLAB

## Requirements
- MATLAB R2022b or later (tested target: R2025a)
- Statistics and Machine Learning Toolbox (required: Modules 2, 3, 5)
- Deep Learning Toolbox (Module 4 only; optional for everything else)
- Check with the MATLAB command `ver` — the two toolboxes must appear in the list.

## Option A — One click (recommended)
1. Unzip `NeuraMedAI-BiomedicalAI-Toolkit-v1.0.0.zip` to a folder with no special
   characters, e.g. `C:\Neura\NeuraMedAI-Toolkit`.
2. Open MATLAB. In the *Current Folder* panel, browse to that folder
   (you should see `RUN_IN_MATLAB.m`, `startup.m`, `src`, `tests`, ...).
3. In the Command Window type:
       RUN_IN_MATLAB
4. Wait for it to finish (1–5 minutes). A file `run_log.txt` appears in the
   toolkit folder. Open it and send me its full contents.

## Option B — Step by step
    cd 'C:\Neura\NeuraMedAI-Toolkit'
    run('startup.m')                 % adds src/ to the path; prints v1.0.0 banner
    cd tests
    run_all_tests                    % runs all 6 test files, prints pass/fail summary
    cd ../examples
    example_disease_classification   % full tabular pipeline with printed metrics
    example_signal_analysis_pipeline % PCA / LDA / model comparison

Run a single test file:
    cd tests
    runtests('test_features_module')

Run one function directly:
    run('startup.m')
    help neuramed.models.trainRandomForest

## Optional — CNN example (Module 4)
Create `examples/sample_images/` with one sub-folder per class
(e.g. `Normal/`, `Pneumonia/`) containing .jpg/.png images, then run
`example_medical_image_classification`. Without that folder it skips itself.

## Reading the results
- `run_all_tests` ends with: Total / Passed / Failed / Incomplete.
- A failing test prints its name, the verification that failed, and a stack line.
- `Incomplete` usually means an error was thrown before the assertion ran.

## Common problems
| Symptom | Fix |
|---|---|
| `Undefined variable "neuramed"` | Run `run('startup.m')` first, or `addpath('<toolkit>\src')` |
| `Undefined function 'fitctree'` (or `fitcsvm`, ...) | Statistics and ML Toolbox not installed |
| `Undefined function 'trainNetwork'` / layer errors | Deep Learning Toolbox not installed |
| `Unrecognized function ... arguments` / syntax error | MATLAB older than R2022b |
| Tests find no files | Current Folder must be `tests` (or run via RUN_IN_MATLAB) |
| Model-save error on a network drive | Copy toolkit to a local drive (-v7.3 MAT files need local disk) |

## What to send back
Paste the full contents of `run_log.txt`. If MATLAB crashed before creating it,
paste the red error text from the Command Window instead.
