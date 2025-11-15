# Infrastructure Fixes - Design Document

**Date:** 2025-11-15
**Status:** Approved Design
**Context:** Fixes for three blockers identified in WIP review (docs/WIP-review.md)

---

## Executive Summary

Address three critical blockers in the Nix infrastructure before wider rollout:
1. Shell hook hides `uv sync` failures
2. Documentation references outdated file paths
3. Workspace bootstrapping is fragile

**Solution:** Remove automatic sync, fix all documentation paths, add clear user guidance.

---

## Design Principles

1. **Explicit over Implicit**: Manual `uv sync` gives users control and visibility
2. **Fast Shell Entry**: Nix shells should be near-instant, no network calls
3. **Clear Error Messages**: Users should see real errors, not generic fallbacks
4. **Accurate Documentation**: Docs must match actual repository structure

---

## Blocker 1: Shell Hook Hides Sync Failures

### Problem

Current implementation in `flake.nix:60-62`:
```nix
# Initialize UV workspace (creates .venv, runs uv sync)
echo "Initializing UV workspace..."
uv sync --quiet 2>/dev/null || echo "Workspace initialized"
```

**Issues:**
- Runs on every `nix develop` (slow, unnecessary network calls)
- Suppresses all errors with `2>/dev/null`
- Shows misleading "Workspace initialized" when sync fails
- Hides lockfile drift, network outages, dependency conflicts

### Solution

Replace auto-sync with conditional warning:

```nix
# Check if workspace is initialized
if [ ! -d .venv ]; then
  if [ ! -d workspace/libs/evio-core ]; then
    echo "⚠️  Workspace members not found. This appears to be initial setup."
    echo "    See docs/setup.md for workspace initialization steps."
  else
    echo "⚠️  First time setup: Run 'uv sync' to initialize workspace"
  fi
fi
```

**Benefits:**
- No hidden failures - users see real errors when they run `uv sync`
- Fast shell entry - no network calls or dependency resolution
- Explicit control - users decide when to sync
- Clear guidance - different messages for different scenarios

### User Workflow

**First time setup:**
```bash
nix develop
# ⚠️  First time setup: Run 'uv sync' to initialize workspace
uv sync
# Shows real errors if any, or succeeds
```

**Daily usage:**
```bash
nix develop  # Fast, no auto-sync
# Work normally
# After git pull with lockfile changes:
uv sync      # Explicit when needed
```

---

## Blocker 2: Documentation References Old Paths

### Problem

`docs/setup.md:48-59` contains examples with wrong paths:
```bash
# Current (WRONG)
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat
```

Actual repository structure uses `evio/` prefix:
```
evio/scripts/play_dat.py
evio/data/fan/fan_const_rpm.dat
```

Following the documentation reproduces "file not found" errors.

### Solution

Update all path references in documentation to match actual structure.

**Files to fix:**

#### 1. docs/setup.md

**Lines 48-59 - Daily Workflow section:**

Before:
```bash
# Run evio legacy demos
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat

# Use shell aliases
run-demo-fan
run-mvp-1
run-mvp-2
```

After:
```bash
# Run evio legacy demos (note: scripts and data are in evio/ directory)
uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat

# Use shell aliases (already use correct paths)
run-demo-fan
run-mvp-1
run-mvp-2
```

**Add new section after line 30 (Quick Start):**

```markdown
### Data Files

Event camera dataset files are located in `evio/data/`:
- `evio/data/fan/*.dat` - Fan rotation datasets (202-489 MB each)
- Large binary files, currently tracked in git for hackathon

**Note:** These files may be gitignored in future. See `evio/data/README.md` for details on obtaining datasets.
```

#### 2. .claude/skills/dev-environment.md

**Lines 32-41 - Examples section:**

Before:
```bash
# Run evio demo
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat
```

After:
```bash
# Run evio demo
uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat
```

**Update Common Mistakes section (line 95):**

Add example:
```markdown
❌ `uv run --package evio python scripts/play_dat.py data/fan/file.dat`
✅ `uv run --package evio python evio/scripts/play_dat.py evio/data/fan/file.dat`
```

#### 3. flake.nix shellHook

**Already fixed** in commit c5aab76 - aliases use correct paths:
```nix
alias run-demo-fan='uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat'
```

No changes needed here.

### Benefits

- Documentation matches actual repository structure
- Users won't encounter "file not found" errors
- Clear note about data file location and size
- Consistent paths across all documentation

---

## Blocker 3: Workspace Bootstrapping Fragile

### Problem

Shell hook creates empty `workspace/{libs,plugins,apps}` directories but:
- If member packages (evio-core, fan-bbox, detector-ui) don't exist, globs are empty
- Auto-sync with empty globs fails
- No clear guidance for users on what to do

### Solution

**Primary fix:** Blocker 1 solution (removing auto-sync) eliminates the fragile auto-sync behavior.

**Secondary fix:** Enhanced shell hook messages provide clear guidance:

```nix
# Check workspace initialization status
if [ ! -d .venv ]; then
  if [ ! -d workspace/libs/evio-core ]; then
    echo "⚠️  Workspace members not found. This appears to be initial setup."
    echo "    See docs/setup.md for workspace initialization steps."
  else
    echo "⚠️  First time setup: Run 'uv sync' to initialize workspace"
  fi
fi
```

**Logic:**
- No `.venv` + No `evio-core` → Incomplete workspace, point to setup docs
- No `.venv` + Has `evio-core` → Normal first run, instruct to sync
- Has `.venv` → Workspace initialized, no message

### Benefits

- No fragile auto-sync that can fail on empty workspace
- Clear differentiation between "incomplete setup" and "just needs sync"
- Users understand what state they're in
- Points to docs for proper initialization

---

## Implementation Changes Summary

### Files to Modify

1. **flake.nix** (lines 60-62)
   - Replace auto-sync with conditional warning
   - Add smart message based on workspace state

2. **docs/setup.md** (multiple sections)
   - Fix all path references (add `evio/` prefix)
   - Add data files section
   - Update workflow to include manual `uv sync` step

3. **`.claude/skills/dev-environment.md`** (examples section)
   - Fix all path references (add `evio/` prefix)
   - Add common mistake example for paths

### Expected Behavior After Fixes

**Scenario 1: Fresh clone, first nix develop**
```bash
$ git clone <repo>
$ cd evio-evlib
$ nix develop
========================================
  Event Camera Detection Workbench
========================================

Python: Python 3.11.14
UV: uv 0.9.7

⚠️  First time setup: Run 'uv sync' to initialize workspace

[... rest of help text ...]

$ uv sync
Resolved 30 packages in 45ms
Installed 22 packages in 1.2s
  + evio-core
  + fan-bbox
  + detector-ui
  + evlib
  [... etc ...]
```

**Scenario 2: Daily usage, workspace already initialized**
```bash
$ nix develop
========================================
  Event Camera Detection Workbench
========================================

Python: Python 3.11.14
UV: uv 0.9.7

[... help text, no warnings ...]

$ # .venv exists, no sync needed, fast entry
```

**Scenario 3: After git pull with lockfile changes**
```bash
$ git pull
# uv.lock changed
$ nix develop  # Fast, no auto-sync
$ uv sync      # Manual sync when ready
Resolved 31 packages in 50ms
Installed 1 package in 200ms
  + new-dependency
```

**Scenario 4: Incomplete workspace (missing members)**
```bash
$ nix develop
========================================
  Event Camera Detection Workbench
========================================

Python: Python 3.11.14
UV: uv 0.9.7

⚠️  Workspace members not found. This appears to be initial setup.
    See docs/setup.md for workspace initialization steps.
```

---

## Testing Checklist

After implementing fixes, verify:

### Test 1: First Time Setup
- [ ] Fresh clone, run `nix develop`
- [ ] Warning appears: "First time setup: Run 'uv sync'"
- [ ] Run `uv sync`
- [ ] All packages install successfully
- [ ] No errors shown

### Test 2: Shell Entry Speed
- [ ] With `.venv` present, `nix develop` completes in <2 seconds
- [ ] No network calls during shell entry
- [ ] No "Initializing UV workspace..." message

### Test 3: Documentation Accuracy
- [ ] All examples in `docs/setup.md` use `evio/scripts/` and `evio/data/` paths
- [ ] All examples in `.claude/skills/dev-environment.md` use correct paths
- [ ] Data files section added to setup docs
- [ ] Demo aliases work as documented

### Test 4: Error Visibility
- [ ] Intentionally break `uv.lock` (corrupt it)
- [ ] Run `uv sync`
- [ ] Real error message appears (not "Workspace initialized")
- [ ] User can see what went wrong

### Test 5: Incomplete Workspace
- [ ] Remove `workspace/libs/evio-core/`
- [ ] Run `nix develop`
- [ ] Warning appears: "Workspace members not found..."
- [ ] Guidance points to setup docs

---

## Migration Notes

**For existing developers:**

After pulling these changes:
1. Exit current `nix develop` shell
2. Re-enter: `nix develop`
3. If you see the warning, run: `uv sync`
4. Workspace is ready

**For new contributors:**

Follow updated `docs/setup.md`:
1. Clone repo
2. `nix develop`
3. `uv sync` (when prompted)
4. Done

---

## Success Criteria

### Must Have
- ✅ Shell hook no longer auto-syncs
- ✅ All documentation paths use `evio/` prefix
- ✅ Clear warning messages for different scenarios
- ✅ `uv sync` errors are visible to users
- ✅ Shell entry is fast (<2 seconds)

### Should Have
- ✅ Data files documented with location and size
- ✅ Common mistakes section updated
- ✅ Migration notes for existing developers

### Nice to Have
- Smart detection of different failure scenarios
- Helpful links in warning messages

---

## Open Questions

**None** - Design is straightforward and addresses all three blockers.

---

## Next Steps

1. Review this design document
2. Create implementation plan
3. Implement fixes in order:
   - Blocker 1 (shell hook)
   - Blocker 2 (documentation)
   - Blocker 3 (already solved by Blocker 1)
4. Test all scenarios
5. Update WIP-review.md to mark blockers as resolved
6. Share updated workflow with other work streams

---

**Ready for implementation plan?**
