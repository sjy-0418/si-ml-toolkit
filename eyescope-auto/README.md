# EyeScope Auto

EyeScope Auto is a Python-based hardware validation automation toolkit for high-speed interface eye-diagram analysis. It loads oscilloscope-style CSV captures, extracts core signal-integrity metrics, applies configurable pass/fail thresholds, diagnoses common failure modes, and generates Markdown validation reports.

The project is designed as a focused portfolio example for UK semiconductor hardware validation roles, including Arm SoC Validation Engineer, Silicon Validation Engineer, Platform Validation Engineer, and Hardware Validation Engineer.

## Why This Matters for Hardware Validation

Modern SoC and platform validation teams need repeatable automation around lab measurements. Eye-diagram checks are commonly used to assess whether a high-speed link has enough voltage and timing margin before deeper debug or compliance work begins.

EyeScope Auto demonstrates the validation workflow expected in a semiconductor lab environment:

- Ingest measured waveform data from test equipment
- Convert raw samples into engineering metrics
- Compare results against validation limits
- Produce a clear pass/fail decision
- Attach likely failure causes to accelerate debug
- Generate an auditable report for bring-up, regression, or issue tracking

## Features

- CSV parser for oscilloscope-style eye-diagram captures
- Eye height calculation at the unit-interval center
- Eye width estimation from threshold crossings
- Jitter estimate from crossing variation
- Configurable validation limits
- Overall PASS/FAIL decision logic
- Rule-based diagnosis for low eye height, narrow eye width, and high jitter
- Markdown report generation with an eye-diagram plot
- Unit tests for metric calculation and pass/fail behavior

## Architecture

```text
eyescope-auto/
├── data/sample_eye.csv          # Example oscilloscope capture
├── eyescope/parser.py           # CSV loading and validation
├── eyescope/metrics.py          # Eye height, eye width, jitter extraction
├── eyescope/mask.py             # Threshold checks and diagnosis rules
├── eyescope/report.py           # Markdown and plot reporting
├── scripts/run_analysis.py      # Command-line validation flow
├── reports/sample_report.md     # Example generated report
└── tests/test_metrics.py        # Unit tests
```

The toolkit separates data ingestion, metric extraction, mask evaluation, and reporting so each stage can be extended independently for lab automation or CI-style regression analysis.

## How to Run

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the sample analysis:

```bash
python scripts/run_analysis.py
```

Run with custom limits:

```bash
python scripts/run_analysis.py --min-eye-height 0.50 --min-eye-width 0.60 --max-jitter 0.025
```

Run tests:

```bash
pytest
```

## Example Output

```text
Overall result: PASS
Eye height: 0.960 V
Eye width: 0.581 UI
Jitter estimate: 0.0080 UI
Report written to: reports/sample_report.md
```

The generated report includes measured values, configured limits, pass/fail status, threshold crossing context, and rule-based diagnosis.

## Engineering Background

This project reflects common validation tasks used during silicon bring-up and platform characterization:

- Automating repetitive oscilloscope measurement review
- Translating waveform captures into margin indicators
- Applying validation masks and acceptance limits
- Creating repeatable evidence for debug and sign-off discussions
- Connecting observed failures to likely signal-integrity, timing, or power-integrity causes

The sample data uses normalized unit interval timing and voltage samples across repeated acquisitions. In a production lab setup, the same structure can be adapted to CSV exports from oscilloscopes, BERTs, compliance tools, or custom measurement scripts.

## Rule-Based Diagnosis

EyeScope Auto reports likely debug directions when a metric fails:

- Low eye height may indicate noise margin degradation, attenuation, or power integrity issue
- Narrow eye width may indicate timing margin issue, inter-symbol interference, or channel loss
- High jitter may indicate clock instability, crosstalk, or signal integrity issue

These rules are intentionally transparent so validation engineers can tune them for a specific PHY, board channel, or interface standard.

## Future Work

- Add support for instrument-specific CSV formats from Tektronix, Keysight, and LeCroy oscilloscopes
- Add mask polygon checking for standard-specific compliance masks
- Export reports to HTML and PDF
- Add batch regression across voltage, temperature, and process corners
- Integrate SCPI-based instrument control for automated lab capture
- Add richer jitter decomposition, including random and deterministic jitter estimates

## Target Roles

This project is relevant to portfolio discussions for:

- Arm SoC Validation Engineer
- Silicon Validation Engineer
- Platform Validation Engineer
- Hardware Validation Engineer
- Signal Integrity Validation Engineer
- Post-silicon Validation Engineer
