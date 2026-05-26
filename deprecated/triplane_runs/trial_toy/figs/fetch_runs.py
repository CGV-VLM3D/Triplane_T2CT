"""Fetch wandb run histories and summary metrics for the three Tier-1 runs."""

import wandb
import json

api = wandb.Api()

run_ids = {
    "trial_toy": "jasonna24-/triplane-ae/shucjhx5",
    "trial_toy_baseline": "jasonna24-/triplane-ae/fs4st1kd",
    "trial_toy_d3t": "jasonna24-/triplane-ae/kp4260wc",
}

all_data = {}

for name, path in run_ids.items():
    print(f"Fetching {name} ({path}) ...")
    run = api.run(path)

    # --- summary / config ---
    summary = dict(run.summary)
    config = dict(run.config)

    # --- full history ---
    # pull all columns we might need
    history = run.history(
        keys=[
            "epoch",
            "val/latent_psnr",
            "val/latent_l1",
            "val/latent_l2",
            "train/loss",
            "val/loss",
        ],
        x_axis="_step",
        pandas=False,  # returns list of dicts
    )

    all_data[name] = {
        "summary": summary,
        "config": config,
        "history": history,
    }

    # print what keys are available in summary
    metric_keys = [k for k in summary.keys() if not k.startswith("_")]
    print(f"  summary keys: {sorted(metric_keys)}")
    print(f"  history rows: {len(history)}")
    if history:
        print(f"  history sample keys: {list(history[0].keys())}")

# Save to disk for the plotting script
with open("/workspace/runs/trial_toy/figs/run_data.json", "w") as f:
    json.dump(all_data, f, indent=2, default=str)

print("\nDone. Saved to run_data.json")
