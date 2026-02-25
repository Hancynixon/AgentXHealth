🩺 AgentXHealth
A Modular, Explainable, and Calibrated Multi-Agent AI Framework for Early Diabetes Risk Prediction
AgentXHealth is a research-grade, clinically grounded AI system for early Type 2 Diabetes Mellitus (T2DM) risk prediction.
Unlike traditional monolithic ML pipelines, AgentXHealth decomposes prediction into domain-specialized intelligent agents and fuses their outputs through a probabilistic reasoning coordinator.
The system integrates:
Physiological modeling (HOMA-IR, QUICKI)
Calibration analysis
Statistical ablation testing
Cross-validation stability analysis
External validation on NHANES
Confidence interval estimation
Explainable AI (SHAP)
Counterfactual reasoning
🔍 Motivation
Most existing diabetes prediction systems:
Use a single monolithic ML model
Focus only on discrimination (AUC)
Ignore probability calibration
Lack external validation
Do not separate clinical domains
Provide only post-hoc feature explanations
AgentXHealth addresses these gaps by introducing:
Explicit domain decomposition
Physiological feature integration
Calibration-aware evaluation
Statistical robustness testing
Cross-cohort validation
Built-in explainability and reasoning
🧠 System Architecture
Copy code

Patient Clinical Data
│
├── LabAgentIntelligent
│     ├── Glucose
│     ├── Insulin
│     ├── HOMA-IR
│     └── QUICKI
│
├── PhysicalAgentIntelligent
│     ├── BMI
│     ├── Blood Pressure
│     └── Clinical staging features
│
├── DemographicAgentIntelligent
│     ├── Age
│     └── Pregnancies (baseline risk)
│
└── CoordinatorReasoner
      ├── Probabilistic fusion
      ├── Dominant-agent detection
      ├── Statistical stacking
      ├── Clinical boost layer
      └── Human-readable explanation

Optional:
└── CounterfactualReasoner
      └── “What-if” intervention simulation
This is not feature concatenation.
It is structured domain-aware reasoning.
🧩 Core Components
🧪 Laboratory Agent
Logistic Regression (class-balanced)
Physiological engineering:
HOMA-IR
QUICKI
Glucose²
Glucose × Insulin
Produces interpretable biochemical risk
🏥 Physical Agent
HistGradientBoostingClassifier
BMI staging (WHO categories)
Blood pressure staging
Captures nonlinear metabolic stress patterns
👥 Demographic Agent
Logistic Regression
Models monotonic baseline risk from age and pregnancies
Provides stable baseline probability scaling
🧠 CoordinatorReasoner
Stacking-based probabilistic fusion
Detects dominant contributing agent
Applies clinical boost under extreme metabolic thresholds
Produces final calibrated risk score
🔄 CounterfactualReasoner
Simulates agent-level feature improvements
Recomputes overall system risk
Ranks intervention impact
Enables actionable “what-if” analysis
📊 Experimental Validation
Internal Validation (Pima Dataset)
Stratified 5-fold Cross-Validation
Mean ROC-AUC = 0.792 ± 0.0378
Hold-out Test Performance:
Accuracy = 0.7100
Precision = 0.5729
Recall = 0.6790
F1 Score = 0.6215
ROC-AUC = 0.7938
95% CI: [0.7315 – 0.8473]
Calibration Analysis
Brier Score = 0.0456
Expected Calibration Error (ECE) = 0.0616
This confirms reasonably reliable probability estimation.
Statistical Ablation Testing
Paired t-tests across model variants confirm:
Removal of Laboratory Agent → significant degradation (p < 0.001)
Removal of Physical/Demographic agents → no significant gain over full model
Architecture justified statistically, not heuristically
External Validation (NHANES Cohort)
Accuracy = 0.9200
ROC-AUC = 0.9606
95% CI: [0.9316 – 0.9828]
Demonstrates strong cross-cohort generalization.
✨ Key Contributions
✔ Modular multi-agent clinical AI architecture
✔ Physiologically grounded feature engineering
✔ Calibration-aware evaluation
✔ Statistical ablation with significance testing
✔ Confidence interval reporting
✔ External cross-population validation
✔ Agent-level SHAP explainability
✔ Counterfactual intervention simulation
📂 Repository Structure
Copy code

AgentXHealth/
├── agents/
│   ├── lab_agent_intelligent.py
│   ├── physical_agent_intelligent.py
│   └── demographic_agent_intelligent.py
│
├── coordinator/
│   ├── coordinator_reasoner.py
│   └── counterfactual_reasoner.py
│
├── evaluation/
│   ├── final_metrics_report.py
│   ├── cv_stability_analysis.py
│   ├── calibration_analysis.py
│   ├── ablation_study.py
│   ├── external_ablation_analysis.py
│   └── nhanes_model_evaluation.py
│
├── explainability/
│   └── shap_nhanes_agents.py
│
├── production/
│   ├── model_runner.py
│   ├── nightly_batch_gsheet.py
│   └── demo_single_case.py
│
├── data/
│   ├── raw/
│   └── NHNES/
│
├── run_agentxhealth.py
└── README.md
▶️ How to Run
Environment Setup
Copy code

conda create -n agentxhealth python=3.10
conda activate agentxhealth
pip install -r requirements.txt
Run Internal Evaluation
Copy code

python -m evaluation.final_metrics_report
Run External Validation
Copy code

python -m evaluation.nhanes_model_evaluation
Run Cross-Validation Stability
Copy code

python -m evaluation.cv_stability_analysis
Run Ablation Study
Copy code

python -m evaluation.ablation_study
Run Full System
Copy code

python run_agentxhealth.py
📌 Research Context
This system was developed as part of an M.Tech (Data Science) research thesis at GITAM University.
The work focuses on:
Modular clinical AI
Interpretable multi-agent reasoning
Statistical rigor in healthcare ML
Cross-cohort validation robustness
⚠️ Disclaimer
AgentXHealth is a research and educational system.
It is not a certified medical diagnostic tool and should not replace professional clinical judgment.
👤 Author
Akoju Kali Eswar Prasad aka Hancy Nixon
M.Tech (Data Science)
AgentXHealth Research Project
