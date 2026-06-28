# Reproducibility Notes

This public portfolio export contains result figures and documentation only. It does not contain the runnable implementation, configuration files, trained models, or final numerical CSV outputs.

The local dissertation notes referenced the following expected implementation structure:

```text
isac_dqn/
|-- environment.py
|-- radio.py
`-- dqn_agent.py
run_experiment.py
results_final/
|-- summary_by_policy.csv
`-- figures/
```

To make this repository fully reproducible later, add:

- source code for the simulation environment and DQN agent,
- dependency file such as `requirements.txt` or `environment.yml`,
- experiment configuration and random seeds,
- final result tables such as `summary_by_policy.csv`,
- scripts that regenerate every figure in `assets/figures/`,
- license information for any adapted open-source framework.

## Expected Rebuild Workflow

After code is added, a practical workflow would be:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_experiment.py
```

The output should regenerate the final result tables and figures used in this portfolio.

## Public Release Caution

Before adding source code, verify the license and attribution requirements of any adapted upstream DQN framework. Do not publish copied or adapted code without a clear compatible license and attribution trail.
