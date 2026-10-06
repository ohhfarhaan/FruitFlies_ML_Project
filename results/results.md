# Experimental Results

## Baseline Pipeline

| Component | Test Result |
|---|---:|
| Fly Count | 0.0% error |
| Male/Female Classification | 0.0% error |
| Male Orientation | 0.0% error |
| Female Orientation | 0.0% error |
| Wing Angle | 0.044 rad (≈2.52°) |

## PCA Comparison for Orientation

| PCA Components | Male Test Accuracy | Female Test Accuracy |
|---:|---:|---:|
| 5 | 98.33% | 98.85% |
| 10 | 99.17% | 98.85% |
| 15 | 99.17% | 98.85% |
| 25 | 99.17% | 98.85% |
| 40 | 100.00% | 98.85% |

## Selected Configuration

We selected 10 PCA components as an efficient configuration because it achieved 99.17% male orientation accuracy and 98.85% female orientation accuracy while using fewer PCA features than the original 15-component configuration.
