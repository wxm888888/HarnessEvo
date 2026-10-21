# Problem-solving skills as a versioned harness component

Problem-solving skills live in the candidate repository at `.opencode/skills/<name>/SKILL.md`, together with optional reference files. They are distinct from `.harness_evolver/skills/`, which guides the evolver itself.

## Data flow

1. The candidate generator may add, edit or delete `.opencode/skills/` files. The existing Git change collector and commit logic include them.
2. Building/registering a binary writes a sibling `opencode.skills.tar.gz`, containing only `.opencode/skills/` files from the exact committed Git tree. Dirty/untracked worktree files do not leak into the package.
3. The sidecar contains a commit identifier, file hashes and skill inventory. It remains separate from the executable, so no OpenCode binary format changes are required.
4. Each OpenSandbox trial passes its local binary sidecar to `OpenCodeCompatHarness`. Setup validates it, uploads it, checks its transfer checksum, and installs it at `/tmp/polar-harness-skills/skills`. The generated OpenCode configuration explicitly includes that path in `skills.paths`.
5. The sampling agent discovers skill descriptions and can load the full text with its skill tool. Installing files proves availability, not that every task will choose to use every skill.
6. Promotion copies the binary and its matching sidecar into the parent-binary directory. The next candidate worktree starts from the accepted Git commit, so the same skill files are present for further evolution.

Deleting the last skill produces an explicit empty package. Setup replaces the dedicated installation directory, preventing deleted skills from lingering. Required missing, mismatched or damaged sidecars fail setup rather than silently running without the intended skills. Legacy standalone binaries remain usable without a sidecar.

This integration currently targets the OpenSandbox evolution backend. It does not silently change the standard TB2.1 runner or the Harbor backend.

## Verification

`harness-evolution/tests/test_solving_skill_pipeline.py` covers committed additions/edits/deletions, future worktree inheritance, actual evolver Git collection/commit, registered binary and promoted-parent sidecars, harness setup/discovery configuration, missing sidecars and symlink rejection.

The live deployment probe is stored outside the repository in the experiment test directory. It uses an existing OpenCode executable to isolate skill deployment from compilation, asks the model to invoke the skill tool, and verifies a random token present only inside the skill. It also reinstalls an empty package and checks the old skill is absent.
