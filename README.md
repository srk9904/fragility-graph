# FragilityGraph

Real-time Structural Debt Prediction using Graph Neural Networks

## Problem Statement

Developers cannot predict which parts of their codebase will break before making changes, leading to fear-driven development that slows feature velocity by 40% in complex projects.

## Solution

FragilityGraph uses Graph Neural Networks to analyze code dependencies and predict the "blast radius" of code changes before they happen. It provides real-time impact analysis as developers type, showing exactly which components will be affected by their changes.

## Key Features

- **Real-time Impact Analysis** - Instant feedback on code changes with sub-second latency
- **Fragility Scoring** - Numerical risk scores (0-100) for every component
- **Blast Radius Visualization** - Interactive graph showing change propagation across services
- **Temporal Learning** - Analyzes Git history to learn which files historically break together
- **IDE Integration** - Seamless VS Code extension for developer workflow

## How It Works

1. Developer types code → Tree-sitter captures AST changes
2. Neo4j graph database updates dependency relationships
3. Graph Neural Network calculates fragility scores
4. Pulse simulation models change propagation
5. Results displayed in IDE with visual blast radius

## Tech Stack

- **Frontend**: TypeScript, VS Code Extension API, Tree-sitter
- **Backend**: Fastify (Node.js), Python FastAPI
- **ML**: PyTorch Geometric, GraphSAGE
- **Database**: Neo4j, Redis

## Documentation

- [Requirements](requirements.md) - Detailed functional and non-functional requirements
- [Design Document](design.md) - System architecture and component design

## What Makes Us Different

Standard AI predicts the next word. FragilityGraph predicts the next break.

Unlike code completion tools, FragilityGraph analyzes structural relationships across your entire repository to provide proactive safety guarantees rather than reactive error detection.

