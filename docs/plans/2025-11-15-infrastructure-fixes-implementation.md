# Infrastructure Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three critical blockers in Nix infrastructure: remove auto-sync, fix documentation paths, improve error messaging.

**Architecture:** Modify flake.nix shell hook to show warnings instead of auto-syncing. Update all documentation to use correct evio/ prefixed paths.

**Tech Stack:** Nix flakes, Markdown documentation

---

## Prerequisites

**Working Directory:** `/Users/laged/Codings/laged/evio-evlib` (main branch)

**Expected State:**
- Current branch: `main`
- Design document exists: `docs/plans/2025-11-15-infrastructure-fixes-design.md`
- Files to modify exist: `flake.nix`, `docs/setup.md`, `.claude/skills/dev-environment.md`

**Verify before starting:**
```bash
git status  # Should show clean or only untracked files
ls flake.nix docs/setup.md .claude/skills/dev-environment.md  # All should exist
```

---

## Task 1: Fix Shell Hook - Remove Auto-Sync

**Goal:** Replace auto-sync in flake.nix with conditional warning message

**Files:**
- Modify: `flake.nix:54-62`

**Step 1: Read current shell hook section**

Run: `cat flake.nix | sed -n '54,62p'`

Expected: Shows current auto-sync code with `uv sync --quiet 2>/dev/null || echo "Workspace initialized"`

**Step 2: Replace auto-sync with conditional warning**

Replace lines 54-62 in `flake.nix`:

**Old code (lines 54-62):**
```nix
            # Create workspace structure if missing
            if [ ! -d workspace ]; then
              echo "Creating workspace structure..."
              mkdir -p workspace/libs workspace/plugins workspace/apps
            fi

            # Initialize UV workspace (creates .venv, runs uv sync)
            echo "Initializing UV workspace..."
            uv sync --quiet 2>/dev/null || echo "Workspace initialized"
```

**New code:**
```nix
            # Create workspace structure if missing
            if [ ! -d workspace ]; then
              echo "Creating workspace structure..."
              mkdir -p workspace/libs workspace/plugins workspace/apps
            fi

            # Check if workspace needs initialization
            if [ ! -d .venv ]; then
              if [ ! -d workspace/libs/evio-core ]; then
                echo "⚠️  Workspace members not found. This appears to be initial setup."
                echo "    See docs/setup.md for workspace initialization steps."
              else
                echo "⚠️  First time setup: Run 'uv sync' to initialize workspace"
              fi
            fi
```

**Step 3: Verify the change**

Run: `cat flake.nix | sed -n '54,67p'`

Expected: Shows new code with conditional warnings, no auto-sync

**Step 4: Test the change**

Run: `nix flake check`

Expected: No errors (flake syntax is valid)

**Step 5: Commit**

```bash
git add flake.nix
git commit -m "fix: remove auto-sync from shell hook, add conditional warnings

- Replace automatic 'uv sync' with warning messages
- Different messages for missing workspace vs missing venv
- Fixes blocker 1: shell hook no longer hides sync failures
- Users now run 'uv sync' explicitly and see real errors"
```

Expected: Commit succeeds

---

## Task 2: Fix Documentation Paths in docs/setup.md

**Goal:** Update all command examples to use correct evio/ prefixed paths

**Files:**
- Modify: `docs/setup.md:48-59`
- Modify: `docs/setup.md` (add data files section after line 30)

**Step 1: Read current examples section**

Run: `cat docs/setup.md | sed -n '46,59p'`

Expected: Shows examples with wrong paths (`scripts/...` and `data/...`)

**Step 2: Fix command examples (lines 48-59)**

Replace the "Running Commands" examples section:

**Old code (lines 48-59):**
```markdown
```bash
# Run evio legacy demos
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat

# Use shell aliases
run-demo-fan
run-mvp-1
run-mvp-2

# Run workspace member code
uv run --package evio-core python -m evio_core
```
```

**New code:**
```markdown
```bash
# Run evio legacy demos (note: scripts and data are in evio/ directory)
uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat

# Use shell aliases (already use correct paths)
run-demo-fan
run-mvp-1
run-mvp-2

# Run workspace member code
uv run --package evio-core python -m evio_core
```
```

**Step 3: Add data files section**

After the "Quick Start" section (around line 30), add new section:

Insert after line 30:

```markdown
### Data Files

Event camera dataset files are located in `evio/data/`:
- `evio/data/fan/*.dat` - Fan rotation datasets (202-489 MB each)
- Large binary files, currently tracked in git for hackathon

**Note:** These files may be gitignored in future. See `evio/data/README.md` for details on obtaining datasets.
```

**Step 4: Update workflow to include manual uv sync**

Find the "Quick Start" section and update the workflow:

**Old (around lines 18-30):**
```markdown
```bash
# Clone repo
git clone <repo-url> evio-evlib
cd evio-evlib

# Enter Nix environment
nix develop

# Workspace auto-initialized by shellHook:
# - Creates workspace/ directory structure
# - Runs uv sync to install all packages
# - Sets up shell aliases
```

That's it! You're ready to develop.
```

**New:**
```markdown
```bash
# Clone repo
git clone <repo-url> evio-evlib
cd evio-evlib

# Enter Nix environment
nix develop

# Initialize workspace (first time only)
uv sync

# Workspace is ready:
# - All packages installed
# - Shell aliases available
# - Ready to run demos
```

You're ready to develop!
```

**Step 5: Verify changes**

Run: `grep -n "evio/scripts" docs/setup.md`

Expected: Shows line numbers where evio/scripts appears (should include our fixes)

Run: `grep -n "Data Files" docs/setup.md`

Expected: Shows the new Data Files section heading

**Step 6: Commit**

```bash
git add docs/setup.md
git commit -m "docs: fix file paths in setup guide

- Update all examples to use evio/ prefix for scripts and data
- Add Data Files section explaining dataset location
- Update Quick Start workflow to include manual 'uv sync' step
- Fixes blocker 2: documentation now matches repository structure"
```

Expected: Commit succeeds

---

## Task 3: Fix Documentation Paths in .claude/skills/dev-environment.md

**Goal:** Update all command examples to use correct evio/ prefixed paths

**Files:**
- Modify: `.claude/skills/dev-environment.md` (Examples section around lines 32-41)
- Modify: `.claude/skills/dev-environment.md` (Common Mistakes section around line 95)

**Step 1: Read current examples section**

Run: `cat .claude/skills/dev-environment.md | sed -n '32,41p'`

Expected: Shows examples with wrong paths

**Step 2: Fix examples section (lines 32-41)**

Replace the examples:

**Old code:**
```markdown
**Examples:**
```bash
# Run evio demo
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat

# Run workspace member script
uv run --package evio-core python -m evio_core.loaders

# Use shell aliases
run-demo-fan
run-mvp-1
```
```

**New code:**
```markdown
**Examples:**
```bash
# Run evio demo
uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat

# Run workspace member script
uv run --package evio-core python -m evio_core.loaders

# Use shell aliases
run-demo-fan
run-mvp-1
```
```

**Step 3: Update Common Mistakes section**

Find the "Common Mistakes" section (around line 95) and add path example:

**Add after existing examples:**
```markdown
❌ `uv run --package evio python scripts/play_dat.py data/fan/file.dat`
✅ `uv run --package evio python evio/scripts/play_dat.py evio/data/fan/file.dat`
```

**Step 4: Update "Never" section (lines 44-46)**

Verify the "Never" section includes path guidance:

**Current (lines 44-46):**
```markdown
**Never:**
- ❌ `cd workspace/libs/evio-core && python ...`
- ❌ `cd evio && uv run ...`
```

**Updated:**
```markdown
**Never:**
- ❌ `cd workspace/libs/evio-core && python ...`
- ❌ `cd evio && uv run ...`
- ❌ `uv run --package evio python scripts/...` (missing evio/ prefix)
```

**Step 5: Verify changes**

Run: `grep -n "evio/scripts" .claude/skills/dev-environment.md`

Expected: Shows updated examples with correct paths

Run: `grep -n "evio/data" .claude/skills/dev-environment.md`

Expected: Shows data path examples

**Step 6: Commit**

```bash
git add .claude/skills/dev-environment.md
git commit -m "docs: fix file paths in dev-environment skill

- Update all examples to use evio/ prefix for scripts and data
- Add common mistake example for incorrect paths
- Add 'Never' example for missing evio/ prefix
- Ensures subagents use correct paths in all commands"
```

Expected: Commit succeeds

---

## Task 4: Test Manual Sync Workflow

**Goal:** Verify the fixed shell hook and workflow work correctly

**Files:**
- None (testing only)

**Step 1: Test with .venv present (normal case)**

Run: `ls .venv/`

Expected: Shows virtual environment exists

Exit and re-enter nix shell:
```bash
exit
nix develop
```

Expected:
- Shell loads quickly (no sync)
- No warning message appears
- Help text displays normally

**Step 2: Test with .venv missing (first time)**

Remove .venv to simulate first-time setup:
```bash
rm -rf .venv
exit
nix develop
```

Expected:
- Shell loads quickly
- Warning appears: "⚠️  First time setup: Run 'uv sync' to initialize workspace"
- Help text displays

**Step 3: Test manual sync**

Run: `uv sync`

Expected:
- Packages resolve and install
- No errors (or real errors are visible, not hidden)
- .venv created successfully

**Step 4: Test sync error visibility**

Corrupt uv.lock to test error visibility:
```bash
echo "CORRUPTED" >> uv.lock
uv sync
```

Expected:
- Real error message appears (parsing error, not "Workspace initialized")
- User can see what went wrong

Restore lockfile:
```bash
git checkout uv.lock
```

**Step 5: Test incomplete workspace scenario**

Remove workspace member to test warning:
```bash
rm -rf workspace/libs/evio-core
rm -rf .venv
exit
nix develop
```

Expected:
- Warning appears: "⚠️  Workspace members not found. This appears to be initial setup."
- Guidance: "See docs/setup.md for workspace initialization steps."

Restore workspace member:
```bash
git checkout workspace/libs/evio-core
uv sync
```

Expected: Workspace restored successfully

**Step 6: Verify no commit needed**

Run: `git status`

Expected: Working tree clean (all test changes were reverted)

---

## Task 5: Update Setup Documentation with Manual Sync Workflow

**Goal:** Ensure docs/setup.md clearly describes the manual sync workflow

**Files:**
- Modify: `docs/setup.md` (Daily Workflow section around lines 39-43)

**Step 1: Read current Daily Workflow section**

Run: `cat docs/setup.md | sed -n '35,80p'`

Expected: Shows "Daily Workflow" section

**Step 2: Add manual sync guidance**

Find the "Enter Environment" subsection and update it:

**Old:**
```markdown
### Enter Environment

```bash
cd evio-evlib
nix develop
```
```

**New:**
```markdown
### Enter Environment

```bash
cd evio-evlib
nix develop
```

**First time setup:**
```bash
# After entering nix shell for the first time:
uv sync  # Install all workspace packages
```

**After git pull with lockfile changes:**
```bash
git pull
nix develop
uv sync  # Update packages if uv.lock changed
```
```

**Step 3: Verify the change**

Run: `grep -A 10 "Enter Environment" docs/setup.md`

Expected: Shows updated section with sync guidance

**Step 4: Commit**

```bash
git add docs/setup.md
git commit -m "docs: add manual sync workflow guidance to setup

- Document first-time setup with explicit 'uv sync' step
- Add guidance for syncing after git pull
- Clarifies when users need to run uv sync manually"
```

Expected: Commit succeeds

---

## Task 6: Final Verification and Summary

**Goal:** Run all verification tests and create summary of changes

**Files:**
- None (testing and documentation only)

**Step 1: Verify all documentation is consistent**

Run verification commands:

```bash
# Check all evio/ prefix usage
grep -r "python scripts/" docs/ .claude/ && echo "FAIL: Found old paths" || echo "PASS: No old paths"

grep -r "python evio/scripts/" docs/ .claude/ && echo "PASS: Found correct paths" || echo "FAIL: Missing correct paths"

# Check shell hook has no auto-sync
grep "uv sync" flake.nix && echo "Found uv sync" || echo "No auto-sync found"
```

Expected:
- No old paths found (PASS)
- Correct paths found (PASS)
- "Found uv sync" only in comments or warning messages, not in auto-execute

**Step 2: Test complete workflow**

Simulate fresh contributor:
```bash
rm -rf .venv
exit
nix develop
# Should see: "⚠️  First time setup: Run 'uv sync' to initialize workspace"
uv sync
# Should succeed with visible progress
run-demo-fan
# Should work (if data files present)
```

Expected: All steps work correctly

**Step 3: Verify git history**

Run: `git log --oneline -6`

Expected: Shows 5 commits:
1. add manual sync workflow guidance
2. fix paths in dev-environment skill
3. fix paths in setup guide
4. remove auto-sync from shell hook
5. add infrastructure fixes design

**Step 4: Create summary of changes**

Print summary:
```bash
echo "Infrastructure Fixes Complete!"
echo ""
echo "Changes Made:"
echo "  1. flake.nix - Removed auto-sync, added conditional warnings"
echo "  2. docs/setup.md - Fixed all paths, added data files section, documented manual sync"
echo "  3. .claude/skills/dev-environment.md - Fixed all paths, added common mistakes"
echo ""
echo "Commits: $(git log --oneline main~5..main | wc -l)"
echo ""
echo "Blockers Resolved:"
echo "  ✅ Blocker 1: Shell hook no longer hides sync failures"
echo "  ✅ Blocker 2: Documentation uses correct evio/ prefixed paths"
echo "  ✅ Blocker 3: Workspace bootstrapping robust (no auto-sync on empty workspace)"
echo ""
echo "Next Steps:"
echo "  - Test with other team members"
echo "  - Update WIP-review.md to mark blockers as resolved"
echo "  - Share updated workflow with other work streams"
```

**Step 5: Check git status**

Run: `git status`

Expected: Clean working tree, all changes committed

---

## Success Criteria

### Must Have ✅
- Shell hook no longer runs `uv sync` automatically
- All documentation paths use `evio/scripts/` and `evio/data/` format
- Clear warning messages for different initialization scenarios
- Manual `uv sync` shows real errors (not hidden)
- Shell entry is fast (<2 seconds)

### Verification Checklist

After implementation, verify:

- [ ] `nix develop` with .venv present: fast, no warnings, no sync
- [ ] `nix develop` without .venv: shows warning, guides to `uv sync`
- [ ] `uv sync` with good lockfile: succeeds, shows progress
- [ ] `uv sync` with bad lockfile: shows real error, not generic message
- [ ] All examples in `docs/setup.md` use `evio/scripts/` paths
- [ ] All examples in `.claude/skills/dev-environment.md` use `evio/scripts/` paths
- [ ] Data files section added to `docs/setup.md`
- [ ] `run-demo-fan` alias works (if data present)
- [ ] Git history shows 5 clean commits
- [ ] Working tree is clean

---

## Rollback Procedure

If something goes wrong:

```bash
# See commits
git log --oneline -10

# Soft reset to before fixes (keep changes)
git reset --soft HEAD~5

# Or hard reset (discard changes)
git reset --hard HEAD~5

# Re-enter shell
exit
nix develop
```

---

## Migration Notes

**For existing developers after pulling these changes:**

1. Exit current `nix develop` shell
2. `git pull` to get the fixes
3. Re-enter: `nix develop`
4. If you see the warning: `uv sync`
5. Done - workspace ready

**For new contributors:**

1. Clone repo
2. `nix develop`
3. `uv sync` (when prompted)
4. Start working

---

**Implementation Time Estimate:** 20-30 minutes

**Tasks:** 6 total
**Commits:** 5 commits
**Files Modified:** 3 files
**Files Created:** 0 files

---

**Ready to execute with superpowers:executing-plans or superpowers:subagent-driven-development**
