#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/push_to_github.sh
#
# Idempotent bootstrap that:
#   1. Initializes a git repo in the project root (if missing)
#   2. Configures a local git identity (user.name, user.email) if missing
#   3. Adds the GitHub remote https://github.com/aayush31bhatt-dev/MLModelsITrained.git
#      (replaces any existing 'origin' pointing elsewhere)
#   4. Stages all files with `git add .` (respecting .gitignore)
#   5. Creates an initial commit with a descriptive message (if there are no commits)
#   6. Pushes to `origin main`, falling back to `master` if `main` fails
#
# Safe to re-run. No secrets are written to disk.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Resolve project root (parent of the scripts/ directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REMOTE_URL="https://github.com/aayush31bhatt-dev/MLModelsITrained.git"
COMMIT_MSG="Initial commit: SMS Spam Classifier (3-model TF-IDF + FastAPI backend)"

# Local-only identity (does NOT touch global git config).
GIT_USER_NAME="aayush31bhatt-dev"
GIT_USER_EMAIL="aayush31bhatt-dev@users.noreply.github.com"

echo "[1/6] Project root: ${PROJECT_ROOT}"

# ── 1. git init (if missing) ────────────────────────────────────────────────
if [ ! -d ".git" ]; then
  echo "[2/6] Initializing git repository ..."
  git init
else
  echo "[2/6] Git repository already initialized."
fi

# ── 2. Configure local identity (if missing) ────────────────────────────────
CURRENT_NAME="$(git config --get user.name || true)"
CURRENT_EMAIL="$(git config --get user.email || true)"

if [ -z "${CURRENT_NAME}" ]; then
  echo "[3/6] Setting local user.name -> ${GIT_USER_NAME}"
  git config user.name "${GIT_USER_NAME}"
else
  echo "[3/6] user.name already set: ${CURRENT_NAME}"
fi

if [ -z "${CURRENT_EMAIL}" ]; then
  echo "[3/6] Setting local user.email -> ${GIT_USER_EMAIL}"
  git config user.email "${GIT_USER_EMAIL}"
else
  echo "[3/6] user.email already set: ${CURRENT_EMAIL}"
fi

# Use 'main' as the default branch name for new repos.
git config init.defaultBranch main || true

# ── 3. Add / replace the GitHub remote ──────────────────────────────────────
if git remote get-url origin >/dev/null 2>&1; then
  EXISTING_URL="$(git remote get-url origin)"
  if [ "${EXISTING_URL}" != "${REMOTE_URL}" ]; then
    echo "[4/6] Replacing existing origin (${EXISTING_URL}) -> ${REMOTE_URL}"
    git remote remove origin
    git remote add origin "${REMOTE_URL}"
  else
    echo "[4/6] Remote 'origin' already points to ${REMOTE_URL}."
  fi
else
  echo "[4/6] Adding remote 'origin' -> ${REMOTE_URL}"
  git remote add origin "${REMOTE_URL}"
fi

# ── 4. Stage all files (respecting .gitignore) ──────────────────────────────
echo "[5/6] Staging files (git add .) ..."
git add .

# Show a short status so the user can sanity-check what will be committed.
echo "----- git status (short) -----"
git status --short
echo "------------------------------"

# ── 5. Commit (only if there is something to commit) ────────────────────────
if git rev-parse HEAD >/dev/null 2>&1; then
  if [ -n "$(git status --porcelain)" ]; then
    echo "[6/6] Existing repo with uncommitted changes — creating commit."
    git commit -m "${COMMIT_MSG}"
  else
    echo "[6/6] Existing repo, working tree clean — skipping commit."
  fi
else
  echo "[6/6] No commits yet — creating initial commit."
  git commit -m "${COMMIT_MSG}"
fi

# ── 6. Push to origin (main with master fallback) ───────────────────────────
echo "[push] Attempting: git push -u origin main"
if git push -u origin main; then
  echo "[push] SUCCESS: pushed to origin main."
else
  echo "[push] 'main' push failed — falling back to 'master'."
  # Rename local branch to master if it isn't already, then push.
  CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if [ "${CURRENT_BRANCH}" != "master" ]; then
    git branch -M master
  fi
  git push -u origin master
  echo "[push] SUCCESS: pushed to origin master."
fi

echo "Done."
