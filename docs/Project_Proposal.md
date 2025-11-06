# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal Template

## 1. Project Title
Simulating Analog Weights for Low-Power AI

List of students in the team:
John Schroter
Matthew Anderson

## 2. Platform Selection
Select one platform category and justify your choice.
Open-source simulator - AIHWKIT.
It lets us study analog in-memory AI without hardware, runs on Colab/laptops, plugs into PyTorch, and is free/reproducible for the class. This matches our goal to compare digital vs. analog trade-offs (accuracy vs. energy) quickly and safely.

## 3. Problem Definition Describe the AI or hardware design problem you aim to address and its relevance to AI hardware (e.g., efficiency, latency, scalability). now give a short reasoning for this
Edge devices (meters, cameras, wearables) have tight power and compute limits. Analog arrays allow for higher compute/byte and fewer data moves which can cut down energy and latency. We can measure the accuracy gap and estimate energy/latency by training a baseline then running the same network in the AIHWKIT simulator. If the trade-off is good, it can support the idea of using analog-style hardware as a good path for low-power edge AI.

## 4. Technical Objectives
List 3–5 measurable objectives with quantitative targets when possible.

Baseline accuracy: Train a Pytorch CNN on MNIST to 99% or higher test accuracy.
Analog accuracy: Map the same model to AIHWKIT and reach 97.5% or higher test accuracy.
Energy benefit: Show an estimated 40% or more lower energy per inference for atleast one analog setting.
Latency: Reduce a simple latency proxy (such as bytes moved or measured batch time) by 20& or more vs. baseline for that setting.
Trade-off study: Produce plot with 3 or more analog configs showing accuracy vs. energy.

Some numbers are subject to change depending on progress and outputs as we begin.

## 5. Methodology
Describe your planned approach: hardware setup, software tools, model design, performance metrics, and validation strategy.

Hardware Setup: None really needed (open-source), run on Google Colab (CPU/GPU), AIHWKIT as simulator.
Software Tools: Pytorch, IBM AIHWKIT, Python, Jupyter/Colab, NumPy/Pandas, Matplotlib.
Model Design: Small CNN for MNIST (baseline), copy same structure into AIHWKIT with different analog settings (precision, noise).
Performance Metrics: Test accuracy, simple energy estimate, latency proxy, model size.
Validation Strategy: Hold-out MNIST test run, run 3 or more analog configurations, compare to baseline, log runs, make accuracy vs. energy plots, possibly confusion matrixes.
## 6. Expected Deliverables
List tangible outputs: working demo, GitHub repository, documentation, presentation slides, and final report.

Working Demo/Notebook (baseline + AIHWKIT runs).
Github Repo (code, configs, results CSVs, figure).
README (how to run, team roles, methods).
Slides (proposal + final results).
Write up and final report (methods, results, plots, takeaways).
## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| [John Schroter] | Analog Sim & Repo Lead | AIHWKIT ports, sweeps, repo/Colab |
| [Matthew Anderson] | ML & Charts Lead | train CNN baseline, metrics, plots |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission, slides guiding process of project |
| 4 | Midterm presentation | Trained CNN in Pytorch with saved metrics and weights |
| 6 | Integration & testing | Same network run through AIHWKIT analog simulator, collect metrics (accuracy, energy estimates, weight noise effects |
| Dec. 18 | Final presentation | Report, demo, GitHub archive, visualizations of accuracy vs. power/latency tradeoff |

## 9. Resources Required
List special hardware, datasets, or compute access needed.

Hardware: None special (open-source), Google Colab (CPU/GPU) or a laptop.
Dataset: MNIST (handwritten digits).
Compute/time: probably between 2 and 4 hours of Colab runtime for training and sim.
Storage: Probably 1-2 GB for notebooks, checkpoints, logs, figures.
Software: Pytorch, IBM AIHWKIT, NumPy/Pandas, Matplotlib, Jupyter/Colab.

## 10. References
Include relevant papers, repositories, and documentation.

IBM Analog Hardware Acceleration Kit (AIHWKIT) - user guide and API docs.
Pytorch - framework docs such as tensors, CNNs, training loop.
Leroux, N., Manea, PP., Sudarshan, C. et al. Analog in-memory computing attention mechanism for fast and energy-efficient large language models. Nat Comput Sci 5, 813–824 (2025). https://doi.org/10.1038/s43588-025-00854-1.
https://research.ibm.com/blog/how-can-analog-in-memory-computing-power-transformer-models.

There will be more references to include as research continues.
