## Workflow to release a new version `vX.Y.Z`

- [ ] Create new branch `vX.Y.Z` from `dev`
- [ ] Bump version in `setup.cfg` and `CADETProcess/__init__.py`
- [ ] Add release notes
  - [ ] General description
  - [ ] Deprecations / other changes
  - [ ] Closed Issues/PRs
  - [ ] Add entry in `index.md`
- [ ] Commit with message `vX.Y.Z`
- [ ] Add tag (`git tag 'vX.Y.Z'`)
- [ ] Push and open PR (base onto `master`). Also push tag: `git push origin --tag`
- [ ] When ready, rebase again onto `dev` (in case changes were made)
- [ ] Merge into master
- [ ] Make release on GitHub using tag and release notes.
- [ ] Check that workflows automatically publish to PyPI and readthedocs
