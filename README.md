# 🛰️ Orbital AI Resilience

## AI Node Health & Autonomous Recovery Engine

A software research prototype exploring an **AI health, trust, and autonomous workload recovery layer** for distributed and future orbital computing environments.

---

## 🚀 What Problem Does It Solve?

A compute node can be physically healthy while the AI computation running on it becomes unreliable.

For example:

* 🌡️ Temperature → Normal
* ⚡ Power → Normal
* 💻 CPU → Normal
* 📡 Communication → Normal
* ❤️ Physical Health → **100%**

But an AI workload may still produce corrupted outputs because of computational faults such as simulated memory bit-flips, model perturbations, or output corruption.

The key question is:

> **Can we determine whether the AI computation itself can still be trusted?**

---

## 🛡️ Core Idea

The system works like an **immune system for AI infrastructure**:

```text
Detect
  ↓
Decide
  ↓
Isolate
  ↓
Route
  ↓
Migrate
  ↓
Execute
  ↓
Verify
  ↓
Recover
```

Instead of only monitoring whether a node is alive, the system evaluates whether its **AI computation remains trustworthy**.

---

## 🧠 Initial Failure Scenario

The primary research scenario is:

### ☢️ Radiation-style Silent AI Degradation

A radiation-style computational fault is **simulated in software** through controlled fault injection.

The simulation can introduce:

* Memory bit flips
* Model parameter perturbation
* Output tensor corruption
* Temperature spikes
* Latency spikes
* Power drain

The current system does **not** simulate physical particle transport or real radiation hardware.

---

## 🔬 Example Detection

In one simulated silent-degradation experiment:

```text
Physical Health        = 100.0%
MSE                    = 0.015288
Safe MSE Threshold     = ≤ 0.005

Cosine Similarity      = 0.6872
Safe Cosine Threshold  = ≥ 0.99
```

The system identifies:

```text
Anomaly            = TRUE
Silent Degradation = TRUE
Physical State     = HEALTHY
Behavioral State   = DEGRADED
```

This demonstrates the central concept:

> **A node can be physically healthy but computationally untrustworthy.**

---

## 🔄 Autonomous Recovery

After detecting degradation, the system can:

1. Detect abnormal AI behavior
2. Evaluate node trust
3. Isolate or quarantine unreliable infrastructure
4. Evaluate healthy target nodes
5. Evaluate communication routes
6. Migrate the workload
7. Execute the workload on the target
8. Independently verify the output
9. Quarantine the target if verification fails
10. Retry using another eligible target
11. Isolate the original degraded source after successful recovery

### Zero-Trust Recovery

Migration alone is **not considered recovery**.

The recovered workload must pass mathematical output verification.

```text
MSE ≤ 0.005
AND
Cosine Similarity ≥ 0.99
```

Verification evidence is recorded using **SHA-256 cryptographic digests**.

---

## 🛰️ Orbital & Constellation Layer

The prototype models orbital-aware recovery using:

* 🌍 Circular Keplerian orbital propagation
* 🌑 Earth-shadow / eclipse modelling
* 📡 3D line-of-sight visibility
* 🔗 Inter-satellite links
* 🧭 Multi-hop Dijkstra routing
* 🔋 Power and eclipse risk
* 📶 Bandwidth-aware transfer evaluation

The recovery engine therefore considers not only **which node is healthy**, but also whether a **valid communication path** exists.

---

## 🤖 Machine Learning

The project uses:

### Isolation Forest

An unsupervised ML detector based on Scikit-learn is used to identify anomalous AI execution feature vectors.

Features include:

* MSE
* MAE
* Cosine similarity
* Maximum output difference
* Mean output difference

The system also uses a statistical **Z-score detector**.

### Important

Reinforcement Learning is **not used as the production decision system**.

The RL component is retained only as an experimental interface/stub.

**Stockfish is not used.**

---

## 🏗️ System Architecture

```text
Physical Telemetry
        │
        ▼
Physical Health Engine
        │
        ├──────────────┐
        ▼              ▼
 AI Workload      Behavioral Metrics
        │              │
        └──────┬───────┘
               ▼
      Anomaly Detection
     Statistical + ML
               │
               ▼
     Autonomous Controller
               │
               ▼
   Target + Route Selection
               │
        ┌──────┴──────┐
        ▼             ▼
   Orbital Model   ISL Routing
        │             │
        └──────┬──────┘
               ▼
       Workload Migration
               │
               ▼
      Independent Verification
               │
       ┌───────┴────────┐
       ▼                ▼
    VERIFIED        FAILED
       │                │
       ▼                ▼
   Recover          Quarantine
```

---

## 📦 Project Structure

```text
orbital-ai-resilience/
│
├── orbital_ai_resilience/
│   ├── core/
│   ├── telemetry/
│   ├── health/
│   ├── faults/
│   ├── workloads/
│   ├── detection/
│   ├── recovery/
│   ├── verification/
│   ├── quarantine/
│   ├── optimization/
│   ├── orbital/
│   ├── constellation/
│   ├── validation/
│   └── dashboard/
│
├── tests/
│
├── complete_validation_report.txt
├── Orbital_AI_Resilience_Viva_Study_Guide.pdf
└── README.md
```

---

## 🧪 Validation

Current validation includes:

* **103 / 103 unit & integration tests passing**
* **23 / 23 autonomous validation cases passing**
* Deterministic reproducibility using seeded experiments
* 50-node constellation scalability testing
* Silent AI degradation testing
* Memory fault injection
* Workload migration
* Target quarantine
* Cascading failure scenarios
* Eclipse and orbital validation
* Multi-hop routing
* Long-duration simulation
* Scientific policy comparison

---

## 📊 Scalability

The simulation has been evaluated with:

| Nodes | Active Crosslinks | Workload Survival |
| ----: | ----------------: | ----------------: |
|     5 |                20 |              100% |
|    10 |                90 |              100% |
|    25 |               600 |              100% |
|    50 |             2,450 |              100% |

The 50-node test completed 20 simulation ticks with approximately **0.00809 seconds per tick** in the reported benchmark.

---

## 🐛 Validation Findings

The validation process was also used to identify system limitations.

### Orbital Route Rejection

In one 5-node single-plane configuration, the tested inter-satellite distance exceeded the configured maximum ISL range.

Instead of forcing an invalid transfer, the orbital policy rejected the route.

This demonstrates an important principle:

> **A resilience system should be able to reject an unsafe recovery rather than report a false success.**

A fixed-metric fallback issue was also identified during validation and corrected so relevant metrics could be derived from migration transaction data.

---

## 🖥️ Dashboard

The project includes a Streamlit dashboard with sections for:

* 🧪 Autonomous Validation
* 🌐 Cluster Topology
* 🛰️ Orbital & Constellation
* 📈 Telemetry & AI Behavior
* 🚨 Silent Degradation & Recovery
* 🛡️ Verification & Quarantine
* ⚡ Adaptive Optimization
* 🔬 Research Benchmarks
* 📜 Event Audit Log
* 🏗️ Architecture & Methodology

---

## ⚙️ Technology Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* Plotly
* Streamlit
* Pytest
* Python Standard Library

---

## ▶️ Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run orbital_ai_resilience/dashboard/app.py
```

Run the test suite:

```bash
pytest
```

---

## 🔐 Scientific Boundaries

This project is a:

**Deterministic software simulation and research prototype.**

It is **not**:

* Flight-certified software
* Physical satellite hardware
* An in-orbit deployment
* A physical radiation test
* A replacement for flight-qualified systems

The orbital environment, radiation-style faults, satellite communication and hardware behavior are represented through software models.

---

## 🔮 Future Research

Possible future extensions include:

* J2 orbital perturbation modelling
* More accurate eclipse geometry
* Advanced radiation models
* Hardware-in-the-loop testing
* FPGA / radiation-tolerant hardware experiments
* Real telemetry integration
* Flight-software adapter layers
* Larger constellation simulations
* Autonomous multi-agent recovery

---

## 🎯 Project Identity

> **Orbital AI Resilience is a software resilience layer that evaluates whether distributed AI computation can still be trusted, detects silent degradation, autonomously recovers workloads, and verifies recovered computation before restoring trust.**

---

## ⚠️ Disclaimer

This project is an independent software research prototype.

It is **not officially endorsed, sponsored, deployed, tested, or adopted by SpaceX, NASA, ISRO, NVIDIA, Intel, Google, or any other aerospace or technology organization.**

---

## 👨‍💻 Project

**Orbital AI Resilience**
**AI Node Health & Autonomous Recovery Engine**

Built as an exploration of autonomous resilience for future distributed AI and orbital computing systems.
