# AgentXHealth 🩺  
### A Multi-Agent, Explainable AI System for Diabetes Risk Prediction

AgentXHealth is an end-to-end, research-grade AI system that predicts diabetes risk using **domain-specialized intelligent agents** and a **reasoning-based coordination layer**.  
Unlike traditional single-model pipelines, AgentXHealth decomposes the prediction task into multiple interpretable agents and reasons over their outputs to deliver **transparent, actionable, and trustworthy decisions**.

---

## 🔍 Motivation

Most existing diabetes prediction systems suffer from one or more of the following limitations:
- Reliance on a **single monolithic model**
- Explainability limited to **post-hoc methods** (e.g., SHAP)
- Lack of **domain separation** in preprocessing and reasoning
- No system-level explanation or “what-if” analysis

**AgentXHealth addresses these gaps** by introducing an explicit **multi-agent architecture** with built-in explainability and counterfactual reasoning.

---

## 🧠 System Architecture

Patient Data
│
├── LabAgentIntelligent
│ └── Laboratory-based risk reasoning
│
├── PhysicalAgentIntelligent
│ └── BMI & blood-pressure-based risk reasoning
│
├── DemographicAgentIntelligent
│ └── Population-level baseline risk
│
└──► CoordinatorReasoner
├── Agent-level arbitration
├── Dominant-agent detection
├── Conflict awareness
└── Human-readable decision explanation
│
└── CounterfactualReasoner
└── System-level “what-if” analysis


---

## 🧩 Core Components

### Intelligent Agents
Each agent:
- Learns an **interpretable risk function**
- Operates on **domain-specific features**
- Outputs a **risk score + explanations**

| Agent | Responsibility |
|------|----------------|
| `LabAgentIntelligent` | Glucose & insulin-based risk |
| `PhysicalAgentIntelligent` | BMI & blood pressure-based risk |
| `DemographicAgentIntelligent` | Age & pregnancy-based baseline risk |

---

### CoordinatorReasoner
- Aggregates **agent-level risk scores**
- Detects **dominant agents**
- Identifies **inter-agent conflicts**
- Produces **human-readable explanations**

This layer performs **reasoning**, not feature concatenation or ensembling.

---

### CounterfactualReasoner
- Simulates **agent-level improvements**
- Recomputes system risk
- Ranks interventions by **risk-reduction impact**

This enables real **“what-if” analysis**, beyond feature importance.

---

## ✨ Key Contributions

- ✔ True **multi-agent ML architecture**
- ✔ Explainability **by design**, not post-hoc
- ✔ Agent-level + system-level reasoning
- ✔ Counterfactual intervention ranking
- ✔ End-to-end working research system

---

## 📂 Repository Structure

AgentXHealth/
├── agents/
│ ├── lab_agent_intelligent.py
│ ├── physical_agent_intelligent.py
│ ├── demographic_agent_intelligent.py
│ └── (baseline agents)
│
├── coordinator/
│ ├── coordinator_reasoner.py
│ ├── counterfactual_reasoner.py
│ └── (baseline coordinator)
│
├── tests/
│ ├── test_physical_agent_intelligent.py
│ ├── test_demographic_agent_intelligent.py
│ ├── test_full_reasoning.py
│ └── test_counterfactual_reasoning.py
│
├── data/
│ └── raw/diabetes.csv
│
├── notebooks/
│ └── (baseline experiments)
│
├── run_agentxhealth.py
└── README.md


---

## ▶️ How to Run

### 1. Environment setup
```bash
conda create -n agentxhealth python=3.10
conda activate agentxhealth
pip install -r requirements.txt
2. Run system tests
python tests/test_full_reasoning.py
python tests/test_counterfactual_reasoning.py
3. Run full pipeline
python run_agentxhealth.py
🧪 Example Output
{
 'final_risk': 0.47,
 'agent_contributions': {
     'lab': 0.31,
     'physical': 0.09,
     'demographic': 0.07
 },
 'dominant_agent': 'lab',
 'conflict_detected': False,
 'decision_explanation': 
     'Laboratory indicators primarily drive overall risk.'
}
📌 Research Context
This project was developed as part of an M.Tech research thesis and is designed to:

Address documented gaps in XAI-based healthcare systems

Demonstrate the effectiveness of agent-based reasoning

Serve as a foundation for future clinical decision-support tools

⚠️ Disclaimer
This system is intended for research and educational purposes only and should not be used as a standalone medical diagnostic tool.

👤 Author
Akoju Kali Eswar Prasad aka Hancy Nixon
M.Tech (Data Science)
AgentXHealth Research Project