# Agent Collaboration Guide

## Purpose

This repo supports the Sensofusion Junction hackathon effort to turn **evio-evlib** into the collaborative event-camera platform described in `docs/architecture.md` and `NEXT_STEPS.md`. We rely on multiple AI agents, so this document defines who does what, their default knowledge sources, and the expectations for working together until `.claude/skills` and `CLAUDE.md` are populated.

## Primary Agents

### Claude (default development agent)
- **Focus:** Day-to-day implementation work across the UV workspace (worktrees, plugins, UI, infra) as described in `README.md` and `docs/architecture.md`.
- **Methods:** Must follow the methodologies to be codified in `CLAUDE.md` and any `.claude/skills` prompts as they come online. Assume those instructions take precedence over ad-hoc guidance.
- **Deliverables:** Plans, code, docs, scripts, and tests within the repo branches/worktrees. Runs most commands, handles integrations, and owns execution of accepted plans.

### Codex (event-camera expert & reviewer)
- **Focus:** Event-based camera expertise for the Sensofusion challenge: evlib integration, detector plugins, adapters, real-time streaming, benchmarking.
- **Responsibilities:**
  - Act as the first-line mentor and subject-matter reviewer for event-camera technology decisions.
  - Perform or delegate research when gaps exist in `docs/` or upstream references (evlib, Metavision, etc.).
  - **Critically review every proposed implementation plan before coding starts.** If a plan is missing, unclear, or risky, Codex must block progress and request revisions.
  - Provide guidance on structuring detectors, adapters, and UV workspace layouts in line with `docs/architecture.md`.
- **Workflow:** Primarily asynchronous reviews. Only takes on implementation work when specifically asked; otherwise prioritizes plan validation, research summaries, and mentorship.

## Review & Workflow Expectations
- **Plan-first rule:** No implementation begins without an explicit plan reviewed by Codex. Plans should reference relevant docs (e.g., `docs/architecture.md`, `NEXT_STEPS.md`, `README.md`), identify affected packages (libs, plugins, apps), and outline testing.
- **Source-of-truth docs:** Until more structure exists, rely on `docs/architecture.md`, `README.md`, `NEXT_STEPS.md`, and any branch-specific docs produced by work streams. Link to these in plans/reviews so decisions remain traceable.
- **.claude/skills & CLAUDE.md:** When these files are introduced, both agents must align their behavior with the methodologies defined there. Update this document to reflect new skills or process changes.
- **Escalation:** If requirements are unclear or new research is needed (e.g., Metavision SDK integration), Codex flags the gap, documents findings, and only then greenlights implementation.

## Keeping This Guide Current
- Update `AGENTS.md` whenever roles, methodologies, or required references change.
- Reference new documentation or skills files as they land so Claude and Codex stay in sync with the evolving workspace strategy.

With this guide, Claude remains the primary builder, Codex safeguards the event-camera vision, and both agents keep plans grounded in the architectural direction laid out in `docs/architecture.md`.
