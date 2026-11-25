# Contributing

## Work one wave at a time

1. Start from an approved issue describing the wave. Run `git switch main` and `git pull --ff-only origin main`, then create a wave-specific branch such as `feat/wave6-storage`.
2. Keep the branch focused on that wave. Open one pull request linked to its issue and describe the implementation, verification, and any known limitations.
3. Run the repository checks before requesting review. The full check is `./scripts/verify.sh`; use `uv run pytest` for the test suite alone. Include the observed results in the pull request.
4. Wait until required checks pass and review feedback is resolved. Do not merge while checks or review are pending.
5. Squash-merge the approved pull request. Then update local `main` with `git pull --ff-only origin main` before starting the next wave on a new issue and branch.

Do not combine multiple waves in one pull request or push directly to `main`.
