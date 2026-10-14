# Study methodology and verification boundaries

The original manuscript describes a five-stage workflow: preprocessing, training, performance evaluation, SHAP-based explainability, and deployment-oriented comparison.

**Dataset:** WUSTL-EHMS-2020 healthcare IoT benchmark.  
**Task:** Binary classification of normal traffic versus FDI attack traffic.  
**Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and MLP.  
**Metrics:** Accuracy, precision, recall, F1, ROC-AUC, MCC, training time, and inference latency.

The paper reports LightGBM as the strongest overall classifier, with a confusion matrix of TN 2847, FP 8, FN 26, TP 383. Reported timing values were measured in the original experimental environment; they should **not** be treated as universal hardware benchmarks.

SHAP analysis in the manuscript emphasizes network and physiological signals. Original SHAP arrays and plots have not been independently recovered for this public repository.

## Limitations

- This repository currently contains **reported** metrics, not a verified end-to-end replication.
- Without original preprocessing scripts, hyperparameters, dataset split, and environment details, it is not possible to independently confirm experimental repeatability.
- A single train/test comparison does not establish performance across hospitals, devices, unseen attack distributions, or deployed hardware.
- False negatives are especially important in safety-critical healthcare security, and the reported confusion matrix includes 26 missed attacks.
- The WUSTL dataset must be obtained from its authorized source under its applicable terms.

## Source

Values were transcribed from the research manuscript *Deployment-Oriented Evaluation of Lightweight Machine Learning Models for False Data Injection Detection in Healthcare IoT*, Tables I–II and Figure 2 description.
