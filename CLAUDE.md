# SI-ML Toolkit — Project Context for Claude

## About the Developer
- Background: Hardware engineer with ~5 years of experience in Signal Integrity, Power Integrity, package, and PCB design at Samsung Electronics.
- ML experience: Beginner. Solid Python basics, no production ML.
- Goal: Build a portfolio project to transition into AI/ML hardware roles at companies like Graphcore, ARM, Cerebras in the UK.

## Project Purpose
An open-source toolkit that applies machine learning to signal integrity analysis: parsing Touchstone files, predicting eye diagram quality, detecting impedance discontinuities in S-parameters.

## How You Should Help Me

### Teaching mode, not autopilot mode
- I am LEARNING ML through this project. I must understand every line of code I commit.
- Never write large chunks of code in one go. Work in small, understandable pieces.
- After writing code, briefly explain WHY you chose this approach over alternatives.

### Code style
- Add comments explaining the "why", not just the "what"
- Use type hints (Python 3.11+)
- Prefer explicit over clever — readability > brevity
- Follow PEP 8, use snake_case

### When I ask for code
1. First, briefly outline the approach in plain English
2. Show the code in small chunks (max ~30 lines per chunk)
3. After each chunk, ask if I want to continue or have questions
4. Suggest one follow-up concept I should learn next

### Domain knowledge — I already know this, don't over-explain
- Signal integrity concepts: S-parameters, eye diagrams, TDR, impedance matching, return loss, insertion loss
- PCB/package design basics
- Touchstone file format

### Domain knowledge — I'm learning this, please explain clearly
- PyTorch internals (autograd, nn.Module, DataLoader)
- ML training loop, loss functions, optimizers
- CNN architectures
- Feature engineering for time-series and frequency-domain data
- Model evaluation metrics

### Project conventions
- Tests: pytest, in `tests/` directory
- Lint: ruff
- All public APIs need docstrings (Google style)
- Commit messages in English, Conventional Commits format (feat:, fix:, docs:, refactor:, test:)

### What NOT to do
- Don't generate the whole project structure at once
- Don't add dependencies without telling me what they do
- Don't write code I haven't asked for
- Don't skip error handling in production code

## Current Phase
Week 1 — Environment setup complete. Now starting to learn scikit-rf basics and creating the first Touchstone parser module.
