# Benchmark verification

All eight automated tests passed, including Streamlit horizon/filter controls and chart/download-button presence. Browser checks confirmed chart/interval rendering and both local CSV download events, plus the deployed forecast CSV download event. Downloaded CSV contents were not independently re-parsed. Tests emitted NumPy datetime deprecation warnings.

Synthetic data only; three expanding-window folds at each horizon. Results do not establish real-world impact.

| Horizon | Holt-Winters WAPE | Weekly naive WAPE | Observed interval coverage |
|---|---:|---:|---:|
| 30 days | 3.53% | 5.02% | 91.11% |
| 60 days | 4.20% | 6.95% | 90.00% |
| 90 days | 6.84% | 7.79% | 67.04% |

Intervals have a nominal 90% level but are approximate residual-bootstrap bands. The 90-day undercoverage means they should not be used as calibrated inventory service-level guarantees. Deployed on Streamlit Community Cloud with Python 3.12: https://demand-forecasting-retail.streamlit.app/
