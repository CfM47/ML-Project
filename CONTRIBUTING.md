# Contributing to ML-Project

Welcome! This document explains how to contribute code to this project, following our workflow and using the provided scripts. Please read carefully before starting.

---

## 1️⃣ Creating an Issue

Whenever a problem, task, or feature arises:

1. Create a GitHub issue in the project.
2. Assign yourself (or the appropriate team member).
3. Add relevant tags. If a suitable tag does not exist, ask for it to be created.
4. You may need to add additional labels later as needed.

---

## 2️⃣ Branching

* Always create a branch **from `dev`** to resolve the issue.
* Branch naming must follow the type of work being done:

```
feat/your-branch-name      # For new features
fix/your-branch-name       # For bug fixes
refactor/your-branch-name  # For refactoring
test/your-branch-name      # For tests
```

* Commit regularly to this branch.

---

## 3️⃣ Updating Your Branch

* To keep your branch updated with the base branch:

```bash
git pull --rebase origin [branch-name]
```

* The most common case is updating from `dev`:

```bash
git pull --rebase origin dev
```

---

## 4️⃣ Squashing Commits (Optional but Recommended)

Before pushing your branch, if you have multiple commits, it is recommended to **squash them into a single commit**:

```bash
git rebase -i HEAD~[n]
```

* Replace `[n]` with the number of commits in your branch.
* In the interactive terminal (GNU Nano by default), each commit starts with `pick`.
* Keep the first commit as `pick` and change the rest to `squash`.
* Save and exit.
* If you’re unsure, watch this 4-minute video: [Interactive Rebase](https://youtu.be/H7RFt0Pxxp8?si=cLKcLh27AMf78s30).

---

## 5️⃣ Pushing and Pull Requests

* Push your branch to GitHub:

```bash
git push -u origin your-branch-name
```

* Open a **Pull Request**.

* At least **two team members must approve** before merging.

* If you make additional changes after pushing, either:

```bash
git commit --amend
# or
git rebase -i HEAD~[n]   # to squash commits again
git push -f
```

---

## 6️⃣ Commit Messages

* All commit messages must be **semantic** and follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

* Example commit types:

```
feat: add new machine learning model
fix: correct data preprocessing step
refactor: improve data loader code
test: add unit tests for model evaluation
docs: update README
chore: update dependencies
```

---

## 7️⃣ Using Project Scripts

We provide a set of commands via `make` that uses `uv` or other packages under the hood to maintain code quality:

| Command            | Description                                |
| ------------------ | -------------------------------------------|
| `make lint`        | Check code style with Ruff                 |
| `make lint-fix`    | Automatically fix lint errors              |
| `make typecheck`   | Run Mypy type checking                     |
| `make test`        | Run Pytest tests                           |
| `make commit`      | creates conventional commits interactively |
| `make check-all`   | Run lint, typecheck, and tests together    |

* We recommend running `make check-all` before committing to ensure everything passes.

---

## 8️⃣ Summary

1. Create an issue.
2. Create a properly named branch from `dev`.
3. Commit your changes.
4. Optionally squash commits.
5. Push branch and open a Pull Request.
6. Ensure semantic commit messages.
7. Use `uv` scripts to maintain code quality.
8. Get approvals and merge.

---

Thanks for contributing and keeping the codebase clean 🤪🍠!
