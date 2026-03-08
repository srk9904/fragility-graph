# Requirements

## Project Overview
FragilityGraph is a real-time structural debt prediction system that uses Graph Neural Networks to predict code breaks before they happen. It analyzes code dependencies and historical patterns to provide developers with instant feedback on the impact of their changes.

## Functional Requirements

### 1. Real-time Code Analysis
- **FR-1.1**: System must capture Abstract Syntax Tree (AST) deltas in real-time as developers type
- **FR-1.2**: Changes must be processed with sub-second latency (< 1 second)
- **FR-1.3**: System must track changes across multiple files simultaneously
- **FR-1.4**: Support for major programming languages (Python, JavaScript, TypeScript, Java, Go)

### 2. Graph Construction and Maintenance
- **FR-2.1**: Automatically construct dependency graphs with nodes representing functions, classes, and modules
- **FR-2.2**: Create edges representing calls, imports, and dependencies
- **FR-2.3**: Update graph structure incrementally as code changes
- **FR-2.4**: Maintain historical co-change patterns from Git history
- **FR-2.5**: Support multi-repository analysis for microservices architectures

### 3. Machine Learning Predictions
- **FR-3.1**: Calculate fragility scores for each code component (0-100 scale)
- **FR-3.2**: Generate blast radius predictions showing affected components
- **FR-3.3**: Provide probability scores for potential regressions (percentage-based)
- **FR-3.4**: Run pulse simulations to model change propagation through the graph
- **FR-3.5**: Learn from historical break patterns to improve predictions over time

### 4. Visualization and Reporting
- **FR-4.1**: Display real-time heatmap of codebase fragility in IDE
- **FR-4.2**: Show interactive blast radius visualization for current changes
- **FR-4.3**: Highlight specific files and lines likely to be affected
- **FR-4.4**: Provide color-coded risk indicators (Red/Yellow/Green)
- **FR-4.5**: Generate detailed impact reports with probability scores

### 5. Developer Workflow Integration
- **FR-5.1**: Integrate as IDE extension (VS Code, JetBrains)
- **FR-5.2**: Provide inline risk indicators in code editor
- **FR-5.3**: Show hover tooltips with dependency chain information
- **FR-5.4**: Display warnings before commits for high-risk changes
- **FR-5.5**: Support manual "check impact" commands for on-demand analysis

### 6. Historical Analysis
- **FR-6.1**: Analyze Git commit history to identify co-change patterns
- **FR-6.2**: Track which files historically break together
- **FR-6.3**: Identify "load-bearing" components with high fragility
- **FR-6.4**: Generate technical debt metrics and trends over time

### 7. Team Collaboration Features
- **FR-7.1**: Provide team-wide codebase heatmap for onboarding
- **FR-7.2**: Display "danger zones" for new team members
- **FR-7.3**: Generate refactor confidence scores for planning purposes
- **FR-7.4**: Create shareable impact analysis reports

## Non-Functional Requirements

### 1. Performance
- **NFR-1.1**: AST delta extraction must complete in < 100ms
- **NFR-1.2**: Graph updates must complete in < 500ms
- **NFR-1.3**: GNN inference must complete in < 1 second
- **NFR-1.4**: Total end-to-end latency must be < 2 seconds
- **NFR-1.5**: System must handle repositories up to 1M lines of code
- **NFR-1.6**: Support concurrent analysis for teams up to 50 developers

### 2. Scalability
- **NFR-2.1**: Graph database must scale to 100K+ nodes
- **NFR-2.2**: Support analysis of monorepos with 10+ microservices
- **NFR-2.3**: Handle Git histories with 10K+ commits
- **NFR-2.4**: Cache frequently accessed predictions in Redis

### 3. Reliability
- **NFR-3.1**: System uptime of 99.5% or higher
- **NFR-3.2**: Graceful degradation if ML model is unavailable
- **NFR-3.3**: Automatic retry logic for failed graph updates
- **NFR-3.4**: Data persistence across system restarts

### 4. Usability
- **NFR-4.1**: IDE extension installation in < 5 minutes
- **NFR-4.2**: Zero configuration for basic usage
- **NFR-4.3**: Clear, actionable error messages
- **NFR-4.4**: Intuitive visual representations of risk
- **NFR-4.5**: Documentation and onboarding tutorials

### 5. Security
- **NFR-5.1**: Secure WebSocket connections (WSS)
- **NFR-5.2**: API authentication and authorization
- **NFR-5.3**: Code never leaves organization's infrastructure (on-premise deployment option)
- **NFR-5.4**: Encrypted storage of sensitive graph data
- **NFR-5.5**: GDPR and data privacy compliance

### 6. Maintainability
- **NFR-6.1**: Modular architecture with clear separation of concerns
- **NFR-6.2**: Comprehensive unit and integration tests (80%+ coverage)
- **NFR-6.3**: Detailed logging and monitoring
- **NFR-6.4**: API versioning for backward compatibility
- **NFR-6.5**: Docker containerization for easy deployment

### 7. Compatibility
- **NFR-7.1**: Support Windows, macOS, and Linux
- **NFR-7.2**: Compatible with Git, GitHub, GitLab, Bitbucket
- **NFR-7.3**: VS Code version 1.70 or higher
- **NFR-7.4**: JetBrains IDEs 2022.1 or higher
- **NFR-7.5**: Node.js 18+ and Python 3.9+

## Technical Requirements

### 1. Frontend / IDE Integration
- Tree-sitter for AST parsing
- VS Code Extension API
- TypeScript/JavaScript (ES2020+)
- WebSocket client for real-time communication

### 2. Backend API
- Fastify (Node.js) for high-performance routing
- WebSocket server for bidirectional communication
- Python FastAPI for ML service endpoints
- Redis for caching and message queuing

### 3. Machine Learning
- PyTorch Geometric for graph neural networks
- GraphSAGE or Relational GCN architecture
- scikit-learn for metrics and evaluation
- GPU support for training (optional but recommended)

### 4. Data Storage
- Neo4j for graph database (Community or Enterprise Edition)
- Redis for in-memory caching
- PostgreSQL for metadata and user settings (optional)

### 5. Infrastructure
- Docker and Docker Compose for containerization
- Kubernetes for orchestration (production deployments)
- Prometheus and Grafana for monitoring
- ELK stack or similar for logging

## Constraints

### 1. Technical Constraints
- Initial version supports static languages with strong type systems
- GNN model requires training data (minimum 100 commits)
- Graph database requires minimum 8GB RAM for medium-sized projects
- GPU recommended but not required for model training

### 2. Resource Constraints
- Development timeline: 6 months for MVP
- Team size: 5-7 developers (2 backend, 2 frontend, 1 ML, 1 DevOps, 1 QA)
- Budget: $205,000 for initial development and first year infrastructure

### 3. Business Constraints
- Must provide value within first week of usage
- Cannot require extensive configuration or setup
- Must work with existing developer workflows
- Pricing model: $50-200 per developer per month (SaaS)

## Acceptance Criteria

### Minimum Viable Product (MVP)
- ✓ Real-time AST delta capture in VS Code
- ✓ Basic graph construction from codebase
- ✓ Simple fragility score calculation
- ✓ Visual blast radius in IDE sidebar
- ✓ Support for Python and JavaScript
- ✓ Basic Git history integration

### Version 1.0
- All MVP features plus:
- ✓ Full GNN-based prediction model
- ✓ Historical co-change pattern analysis
- ✓ Interactive heatmap visualization
- ✓ Support for 5+ programming languages
- ✓ Team collaboration features
- ✓ Cloud deployment option

### Version 2.0 (Future)
- Multi-IDE support (VS Code, JetBrains, Vim)
- CI/CD pipeline integration
- Automated refactoring suggestions
- Advanced analytics dashboard
- Enterprise SSO and access controls
- Custom model training for specific codebases

## Dependencies

### External Services
- GitHub/GitLab API for repository access
- Cloud infrastructure (AWS, GCP, or Azure)
- Neo4j AuraDB (managed graph database option)
- Redis Cloud (managed cache option)

### Third-party Libraries
- Tree-sitter language parsers
- PyTorch and PyTorch Geometric
- Neo4j drivers
- React Icons (for UI)
- Chart.js (for visualizations)

## Risk Analysis

### Technical Risks
1. **Graph complexity growth**: Large codebases may create graphs too large to process efficiently
   - Mitigation: Implement graph pruning and sampling strategies
   
2. **False positives**: Model may predict breaks that don't occur
   - Mitigation: Continuous model retraining with feedback loop
   
3. **Language support**: AST parsing differs significantly across languages
   - Mitigation: Start with 2-3 languages, expand gradually

### Operational Risks
1. **Developer adoption**: Developers may ignore predictions if not accurate
   - Mitigation: Start with high-confidence predictions only, improve over time
   
2. **Performance impact**: Real-time analysis may slow down IDE
   - Mitigation: Asynchronous processing, aggressive caching

### Business Risks
1. **Market competition**: Existing code analysis tools may add similar features
   - Mitigation: Focus on GNN differentiator, build strong IP position
   
2. **Data privacy concerns**: Enterprises may not want code leaving their infrastructure
   - Mitigation: Offer on-premise deployment option from day one

## Future Enhancements

### Phase 2 Features
- Automated refactoring suggestions based on fragility analysis
- Integration with project management tools (Jira, Linear)
- Slack/Teams notifications for high-risk commits
- Mobile app for monitoring team metrics

### Phase 3 Features
- AI-powered code review comments
- Predictive test prioritization
- Architecture quality scoring
- Custom GNN model training per organization

## Glossary

- **AST (Abstract Syntax Tree)**: Tree representation of source code structure
- **Blast Radius**: Set of components potentially affected by a code change
- **Co-change Pattern**: Files that frequently change together in commits
- **Fragility Score**: Numerical measure (0-100) of component's likelihood to cause breaks
- **GNN (Graph Neural Network)**: Neural network designed to operate on graph-structured data
- **Load-bearing Component**: Critical code with high coupling and many dependents
- **Pulse Simulation**: Process of injecting a signal at a change point and observing propagation
- **Structural Debt**: Accumulated complexity making refactoring risky and time-consuming