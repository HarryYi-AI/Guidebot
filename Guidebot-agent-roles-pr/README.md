# Guidebot Agent Roles PR Package

This folder contains the code package for the local commit:

```text
adc89b1 Add embodied planner and skill evolution agents
```

## Contents

```text
files/
  src/guidebot/agents/__init__.py
  src/guidebot/agents/embodied_planner.py
  src/guidebot/agents/skill_evolution.py
  tests/test_embodied_planner_agent.py
  tests/test_skill_evolution_agent.py

patches/
  0001-Add-embodied-planner-and-skill-evolution-agents.patch
```

## Option A: Apply as a Git Commit

From a clean Guidebot repository:

```bash
git checkout -b feature/agent-roles
git am /home2/ss/TRAE/Guidebot-agent-roles-pr/patches/0001-Add-embodied-planner-and-skill-evolution-agents.patch
```

Then push and open a pull request:

```bash
git push -u origin feature/agent-roles
```

## Option B: Copy Files Manually

Copy everything under `files/` into the root of a Guidebot repository, preserving
the directory structure.

## Verification Used

```bash
env PYTHONPATH=src python -m compileall src tests
```
