# third_party Submodule Pins

Pinned at Phase A Day 1 (2026-05-26). SHAs are "latest at clone time".
If upstream HEAD has broken in the intervening days, document and re-pin.

| Submodule | Path | URL | Pinned SHA |
|---|---|---|---|
| Report2CT | `third_party/report2ct` | https://github.com/sinaamirrajab/report2ct | `7b483a856ef159cfd0dada249b110d8f8eebf502` |
| GenerateCT | `third_party/generatect` | https://github.com/ibrahimethemhamamci/GenerateCT | `2a811356de351c67f89b2929c8bc9f2390797d9c` |
| VLM3D-Dockers | `third_party/vlm3d_dockers` | https://github.com/forithmus/VLM3D-Dockers | `c73fe07308c9393c94f412a57e874d25792813bf` |

## Pin / restore

Restore exact pinned SHA after `git submodule update --init`:

```bash
git -C third_party/report2ct       checkout 7b483a856ef159cfd0dada249b110d8f8eebf502
git -C third_party/generatect      checkout 2a811356de351c67f89b2929c8bc9f2390797d9c
git -C third_party/vlm3d_dockers   checkout c73fe07308c9393c94f412a57e874d25792813bf
```

## Policy

- third_party/ is **read-only** (project Principle P2). Code adaptation happens in
  `src/baselines/*_adapter.py` (LightningModule wrappers) and `src/eval/vlm3d_runner.py`,
  not by modifying submodule sources.
- Re-pin only after a deliberate upstream sync; update this file in the same commit.
