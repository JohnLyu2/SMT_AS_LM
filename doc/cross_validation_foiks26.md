## Cross Validation Results for FoIKS26

The table below shows the 4-fold cross validation results for each logic. The metric is the PAR-2 VBS-SBS gap (%) closed by the algorithm selector (the same as used in the original submission).

| logic | synt | desc | synt+desc |
| --- | --- | --- | --- |
| ABV | 79.2 ± 9.3 | -0.7 ± 1.2 | 71.2 ± 8.9 |
| ALIA | 60.3 ± 6.7 | 6.9 ± 10.0 | 50.0 ± 7.1 |
| BV | 24.4 ± 7.3 | 31.0 ± 15.0 | 44.6 ± 18.1 |
| QF_IDL | 59.6 ± 11.6 | 40.8 ± 29.2 | 68.4 ± 10.1 |
| QF_LIA | 87.9 ± 2.4 | 85.5 ± 2.9 | 90.5 ± 2.2 |
| QF_NRA | 35.1 ± 11.9 | 37.2 ± 16.6 | 46.7 ± 12.6 |
| QF_SLIA | 91.1 ± 2.6 | 86.6 ± 2.9 | 93.9 ± 1.6 |
| UFLIA | 10.0 ± 24.7 | -19.8 ± 20.7 | 18.2 ± 22.2 |
| UFNIA | 26.4 ± 11.4 | 0.0 ± 0.0 | 32.7 ± 10.7 |

The general trend observed in the original submission remains consistent under the new cross-validation results. Specifically, the description feature alone typically performs worse than the syntactic feature alone, while the combined feature usually outperforms both. In particular, the combined feature outperforms the syntactic feature alone in 7 out of 9 cases, with the exceptions being ABV and ALIA.

A closer examination of the ABV and ALIA logics reveals that there is very little variability in their natural-language descriptions. The SMT-LIB native descriptions are usually identical within a family. Each of these two logics only contains only two families, and in both cases, a single family accounts for more than 95 percent of all benchmarks. Under such conditions, description embeddings are not expected to provide useful discriminative information. In contrast, the other logics have greater variability in their descriptions.

Note: The difference from the results reported in [experiment0103.md](experiment0103.md) is that results presented here are based on 4-fold cross validation over the full dataset for each logic, whereas `experiment0103.md` used 4-fold cross validation on only a subset of the data (80%). Thus,the results reported here are relatively more reliable.
