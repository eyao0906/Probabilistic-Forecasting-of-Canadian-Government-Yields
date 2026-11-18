# Previous experiment cleanup inventory

The new branch starts from `b5bf8ac3783876aaccb616b30ed398fdb2aa40e0`.
The following four directories contain 121 tracked files exclusively belonging
to the previous Hautsch–Yang implementation and are removed from this branch:

- `src/ssdns_hy/`: superseded implementation and synthetic-screen runners.
- `tests/ssdns_hy/`: tests tied to the abandoned implementation.
- `results/hautsch_yang_state_space_20261117/`: rejected screens, trace
  checkpoints, old configuration and diagnostics.
- `docs/hautsch_yang_state_space_20261117/`: documentation for that workflow.

The files and failed outcomes remain available in the unchanged
`research/hautsch-yang-state-space-20261117` branch. A tracked-tree search
found no references to these directories outside them. No dataset, manuscript,
accepted benchmark, historical research archive, or literature source is
included in this cleanup.
