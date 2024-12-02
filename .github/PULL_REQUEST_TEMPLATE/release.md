## Workflow to release a new version `vX.Y.Z`
This is a checklist for a new release.
It assumes a - [ ] Push and open PR (base onto `master`). Also push tag: `git push origin --tag`

- [ ] Create new branch `vX.Y.Z` from `dev`
- [ ] Bump version in `CADETProcess/__init__.py` and `.zenodo.json`
- [ ] Add release notes in `docs/source/release_notes/vX.Y.Z.md`
  - [ ] General description
  - [ ] Deprecations / other changes
  - [ ] Closed Issues / PRs
  - [ ] Add entry in `index.md`
- [ ] When ready, rebase again onto `dev` (in case changes were made)
- [ ] Commit with message `vX.Y.Z`
- [ ] Add tag (`git tag 'vX.Y.Z'`) and push tag: (`git push origin --tag`)
- [ ] Merge into master
- [ ] Make release on GitHub using tag and release notes.
- [ ] Check that workflows automatically publish to PyPI and [https://cadet-process.readthedocs.io/]
