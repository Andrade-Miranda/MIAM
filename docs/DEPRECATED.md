# Local-Only And Deprecated Files

Deprecated implementations and later project-specific experiments may remain
on a developer's workstation for reference, but they are not part of the
paper-focused public repository. Git ignore rules prevent these files from
being added again after they are removed from the index.

## Local-Only Categories

- `models/deprecated/`, `util/deprecated/`, and `Zdeprecated/`.
- `models/ToCheck/`, `config/Tocheck/`, and `data/ToCheck/` prototypes.
- Later project-specific entry points, configurations, evaluators, helpers, and
  orchestration files.
- `nnUNet/data/`, `options/CT/`, `pretrained_ckpt/`, `checkpoints/`, and
  `Output/` data or binary artifacts.
- Local virtual environments, caches, IDE files, and generated package
  metadata.

Ignoring a path does not delete it. If a file was previously tracked, it must be
removed from Git's index while retaining the working-tree copy. Developers can
still inspect and run local-only files, but changes to them are not included in
the paper-focused source history.

Code should move out of the local archive only when it has a documented purpose,
portable configuration, dependency coverage, tests, and a clear relationship to
the public MIAM workflow.
