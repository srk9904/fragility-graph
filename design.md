# Design Document

## 1. System Overview

FragilityGraph is a real-time code analysis system that predicts structural failures before they occur. It combines Abstract Syntax Tree (AST) parsing, graph database technology, and Graph Neural Networks (GNNs) to provide developers with instant feedback on the potential impact of their code changes.

### 1.1 Design Philosophy

**Proactive over Reactive**: Predict problems before they manifest, rather than detecting them after failures.

**Developer-Centric**: Seamlessly integrate into existing workflows without requiring behavior changes.

**Graph-First Thinking**: Model code as a living, interconnected graph rather than isolated files.

**Temporal Intelligence**: Learn from history to predict future patterns.

## 2. Architecture Overview

FragilityGraph follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                          │
│              (IDE Extension + UI)                        │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                    Sensor Layer                          │
│         (Tree-sitter Parser + AST Extractor)            │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                  Communication Layer                     │
│            (Fastify API + WebSocket Server)             │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                   Processing Layer                       │
│         (Task Queue + Delta Processor)                  │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                    Storage Layer                         │
│        (Neo4j Graph DB + Redis Cache + Git)             │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                    ML Engine Layer                       │
│    (PyTorch Geometric + GraphSAGE + Training)           │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                   Analysis Layer                         │
│  (Fragility Calculator + Pulse Simulator + Ranker)      │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│                    Output Layer                          │
│       (Visualization + Heatmap + Risk Reports)          │
└─────────────────────────────────────────────────────────┘
```

## 3. Component Design

### 3.1 Client Layer (IDE Extension)

**Technology**: TypeScript, VS Code Extension API

**Responsibilities**:
- Monitor file changes in the editor
- Display visualization sidebar
- Render heatmaps and blast radius graphs
- Show inline risk indicators
- Handle user interactions

**Key Classes**:
```typescript
class FragilityGraphExtension {
  private treeParser: TreeSitterParser;
  private wsClient: WebSocketClient;
  private visualizer: CodeVisualizer;
  
  activate(): void
  deactivate(): void
  onDocumentChange(delta: TextDocumentChangeEvent): void
  updateVisualization(data: ImpactData): void
}

class CodeVisualizer {
  renderHeatmap(scores: Map<string, number>): void
  renderBlastRadius(graph: DependencyGraph): void
  showInlineRisk(location: CodeLocation, risk: number): void
}
```

### 3.2 Sensor Layer (AST Parser)

**Technology**: Tree-sitter, Node.js

**Responsibilities**:
- Parse source code into Abstract Syntax Trees
- Extract incremental deltas on every keystroke
- Identify functions, classes, imports, and calls
- Send structured deltas to backend

**Key Classes**:
```typescript
class TreeSitterParser {
  private parsers: Map<string, Parser>;
  
  parse(code: string, language: string): Tree
  getDelta(oldTree: Tree, newTree: Tree): ASTDelta
  extractNodes(tree: Tree): CodeNode[]
  extractEdges(tree: Tree): CodeEdge[]
}

interface ASTDelta {
  added: CodeNode[];
  removed: CodeNode[];
  modified: CodeNode[];
  timestamp: number;
}
```

### 3.3 Communication Layer (API Server)

**Technology**: Fastify (Node.js), WebSocket

**Responsibilities**:
- Handle HTTP REST API requests
- Maintain persistent WebSocket connections
- Route messages between clients and processing layer
- Implement authentication and rate limiting

**Key Routes**:
```
POST   /api/v1/analyze           - Submit code change for analysis
GET    /api/v1/fragility/:file   - Get fragility score for file
GET    /api/v1/heatmap/:repo     - Get repository heatmap
WS     /ws                       - WebSocket for real-time updates
```

**Key Classes**:
```javascript
class APIServer {
  private app: FastifyInstance;
  private wsManager: WebSocketManager;
  
  setupRoutes(): void
  handleAnalysisRequest(req, res): Promise<void>
  broadcastUpdate(data: UpdateData): void
}

class WebSocketManager {
  private connections: Map<string, WebSocket>;
  
  addConnection(clientId: string, ws: WebSocket): void
  sendToClient(clientId: string, message: Message): void
  broadcast(message: Message): void
}
```

### 3.4 Processing Layer (Delta Processor)

**Technology**: Python, Redis Queue

**Responsibilities**:
- Receive AST deltas from API layer
- Queue processing tasks
- Coordinate graph updates
- Trigger ML inference

**Key Classes**:
```python
class DeltaProcessor:
    def __init__(self, graph_store, ml_engine):
        self.graph_store = graph_store
        self.ml_engine = ml_engine
        self.queue = RedisQueue()
    
    async def process_delta(self, delta: ASTDelta) -> ImpactReport:
        # Update graph
        await self.graph_store.apply_delta(delta)
        
        # Run ML inference
        predictions = await self.ml_engine.predict(delta.affected_nodes)
        
        # Generate report
        return self.create_impact_report(predictions)
```

### 3.5 Storage Layer (Graph Database)

**Technology**: Neo4j, Redis

**Graph Schema**:
```cypher
// Nodes
(:Function {name, file, lineStart, lineEnd, complexity, fragility})
(:Class {name, file, lineStart, lineEnd, fragility})
(:Module {name, path, fragility})
(:File {path, language, lastModified})

// Relationships
(:Function)-[:CALLS]->(:Function)
(:Function)-[:BELONGS_TO]->(:Class)
(:Class)-[:DEFINED_IN]->(:File)
(:File)-[:IMPORTS]->(:File)
(:File)-[:CO_CHANGES_WITH {frequency, lastCoChange}]->(:File)
```

**Key Queries**:
```cypher
// Find all dependencies of a function
MATCH (f:Function {name: $funcName})-[:CALLS*1..5]->(dep:Function)
RETURN DISTINCT dep

// Find co-change patterns
MATCH (f1:File)-[r:CO_CHANGES_WITH]->(f2:File)
WHERE r.frequency > $threshold
RETURN f1, f2, r.frequency
ORDER BY r.frequency DESC

// Calculate node importance (PageRank-style)
CALL gds.pageRank.stream({
  nodeProjection: 'Function',
  relationshipProjection: 'CALLS'
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS function, score
ORDER BY score DESC
```

**Key Classes**:
```python
class GraphStore:
    def __init__(self, neo4j_uri, credentials):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=credentials)
    
    async def add_node(self, node: CodeNode) -> str:
        pass
    
    async def add_edge(self, edge: CodeEdge) -> None:
        pass
    
    async def get_dependencies(self, node_id: str, depth: int) -> List[CodeNode]:
        pass
    
    async def calculate_importance(self) -> Dict[str, float]:
        pass
```

### 3.6 ML Engine Layer (Graph Neural Network)

**Technology**: PyTorch Geometric, GraphSAGE

**Model Architecture**:
```python
class FragilityGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index):
        # Node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # Fragility prediction
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)  # Output: [0, 1] fragility score
```

**Node Features**:
- Cyclomatic complexity
- Number of incoming edges (callers)
- Number of outgoing edges (callees)
- Historical change frequency
- Lines of code
- Depth in call graph
- Co-change frequency with other nodes

**Training Process**:
```python
class GNNTrainer:
    def __init__(self, model, graph_store):
        self.model = model
        self.graph_store = graph_store
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    def prepare_training_data(self) -> Data:
        # Extract graph from Neo4j
        nodes, edges = self.graph_store.export_graph()
        
        # Create feature matrix
        x = self.create_feature_matrix(nodes)
        
        # Create labels from historical failures
        y = self.extract_failure_labels(nodes)
        
        return Data(x=x, edge_index=edges, y=y)
    
    def train_epoch(self, data: Data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(out, data.y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 3.7 Analysis Layer (Fragility Calculator)

**Technology**: Python, NumPy

**Key Algorithms**:

**1. Fragility Score Calculation**:
```python
def calculate_fragility_score(node: CodeNode, graph: Graph) -> float:
    """
    Fragility = f(complexity, coupling, historical_failures, importance)
    """
    # Base score from GNN
    gnn_score = self.model.predict(node)
    
    # Structural metrics
    coupling = len(graph.get_neighbors(node)) / graph.total_nodes
    complexity = node.cyclomatic_complexity / 20.0  # Normalize
    
    # Historical metrics
    change_frequency = node.change_count / graph.total_commits
    failure_rate = node.failure_count / max(node.change_count, 1)
    
    # Weighted combination
    fragility = (
        0.4 * gnn_score +
        0.2 * coupling +
        0.2 * complexity +
        0.1 * change_frequency +
        0.1 * failure_rate
    )
    
    return min(fragility * 100, 100)  # Scale to 0-100
```

**2. Pulse Simulation**:
```python
class PulseSimulator:
    def simulate_impact(self, change_node: CodeNode, graph: Graph) -> BlastRadius:
        """
        Inject a 'stress signal' and observe propagation.
        """
        visited = set()
        impact_scores = {}
        queue = [(change_node, 1.0)]  # (node, propagation_strength)
        
        while queue:
            current, strength = queue.pop(0)
            
            if current in visited or strength < 0.1:  # Propagation threshold
                continue
            
            visited.add(current)
            impact_scores[current] = strength
            
            # Propagate to neighbors with decay
            for neighbor in graph.get_neighbors(current):
                decay = self.calculate_decay(current, neighbor)
                new_strength = strength * decay
                queue.append((neighbor, new_strength))
        
        return BlastRadius(
            affected_nodes=list(visited),
            impact_scores=impact_scores,
            total_risk=sum(impact_scores.values())
        )
    
    def calculate_decay(self, source: CodeNode, target: CodeNode) -> float:
        """
        Decay based on relationship type and node fragility.
        """
        edge_type = self.graph.get_edge_type(source, target)
        
        if edge_type == "CALLS":
            decay = 0.8  # Strong propagation
        elif edge_type == "IMPORTS":
            decay = 0.6  # Medium propagation
        elif edge_type == "CO_CHANGES_WITH":
            decay = 0.4  # Weak propagation
        else:
            decay = 0.2
        
        # Adjust by target fragility
        target_fragility = target.fragility_score / 100.0
        return decay * (0.5 + 0.5 * target_fragility)
```

**3. Blast Radius Calculation**:
```python
def calculate_blast_radius(change: CodeChange, graph: Graph) -> BlastRadiusReport:
    """
    Determine all components affected by a change.
    """
    # Run pulse simulation
    pulse_result = pulse_simulator.simulate_impact(change.node, graph)
    
    # Categorize by risk level
    high_risk = [n for n, score in pulse_result.impact_scores.items() if score > 0.7]
    medium_risk = [n for n, score in pulse_result.impact_scores.items() if 0.4 < score <= 0.7]
    low_risk = [n for n, score in pulse_result.impact_scores.items() if 0.1 < score <= 0.4]
    
    # Calculate regression probability
    regression_probability = calculate_regression_probability(high_risk, graph)
    
    return BlastRadiusReport(
        high_risk_nodes=high_risk,
        medium_risk_nodes=medium_risk,
        low_risk_nodes=low_risk,
        regression_probability=regression_probability,
        affected_services=identify_affected_services(pulse_result.affected_nodes)
    )
```

### 3.8 Output Layer (Visualization)

**Technology**: React, D3.js, Chart.js

**Key Visualizations**:

**1. Heatmap Generator**:
```typescript
class HeatmapGenerator {
  generateFileHeatmap(repository: Repository): HeatmapData {
    const files = repository.getAllFiles();
    const scores = files.map(f => ({
      path: f.path,
      fragility: f.fragilityScore,
      color: this.getColorForScore(f.fragilityScore)
    }));
    
    return {
      type: 'treemap',
      data: this.organizeByDirectory(scores)
    };
  }
  
  private getColorForScore(score: number): string {
    if (score > 70) return '#F96167';      // Red
    if (score > 40) return '#FFB84D';      // Yellow
    return '#02C39A';                      // Green
  }
}
```

**2. Blast Radius Visualizer**:
```typescript
class BlastRadiusVisualizer {
  renderGraph(blastRadius: BlastRadiusReport): void {
    const nodes = blastRadius.allAffectedNodes.map(n => ({
      id: n.id,
      label: n.name,
      size: n.impactScore * 10,
      color: this.getRiskColor(n.riskLevel)
    }));
    
    const edges = this.extractEdges(blastRadius);
    
    // Render using D3.js force-directed graph
    this.d3Renderer.render({nodes, edges});
  }
}
```

## 4. Data Flow

### 4.1 Real-time Analysis Flow

```
1. Developer types in IDE
   ↓
2. Tree-sitter captures AST delta (< 100ms)
   ↓
3. Delta sent via WebSocket to API server
   ↓
4. API server enqueues processing task
   ↓
5. Delta processor updates Neo4j graph (< 500ms)
   ↓
6. GNN model runs inference on affected nodes (< 1s)
   ↓
7. Pulse simulator calculates blast radius
   ↓
8. Results formatted and sent back via WebSocket
   ↓
9. IDE updates visualization (< 100ms)
   ↓
TOTAL: < 2 seconds end-to-end
```

### 4.2 Historical Analysis Flow

```
1. System periodically analyzes Git history
   ↓
2. Extract commits and changed files
   ↓
3. Identify co-change patterns (files changed together)
   ↓
4. Create CO_CHANGES_WITH relationships in Neo4j
   ↓
5. Mark commits that introduced bugs/failures
   ↓
6. Use failure data to train GNN model
   ↓
7. Update fragility scores based on new patterns
```

## 5. Design Patterns

### 5.1 Observer Pattern
IDE extension observes file changes and notifies the backend via WebSocket.

### 5.2 Strategy Pattern
Different parsing strategies for different programming languages (Python, JavaScript, Java, etc.).

### 5.3 Factory Pattern
Create appropriate node and edge objects based on AST node types.

### 5.4 Singleton Pattern
Single WebSocket connection manager per IDE instance.

### 5.5 Repository Pattern
Abstract data access to Neo4j and Redis behind repository interfaces.

## 6. Security Design

### 6.1 Authentication
- API key authentication for IDE extensions
- JWT tokens for web-based access
- OAuth 2.0 integration with GitHub/GitLab

### 6.2 Authorization
- Role-based access control (Developer, Tech Lead, Admin)
- Repository-level permissions
- Team-based data isolation

### 6.3 Data Privacy
- Code never leaves organization's infrastructure (on-premise option)
- Encrypted WebSocket connections (WSS)
- Encrypted at rest in Neo4j and Redis
- GDPR-compliant data retention policies

### 6.4 Rate Limiting
- 100 requests per minute per API key
- 10 concurrent WebSocket connections per user
- Graceful degradation under heavy load

## 7. Performance Optimization

### 7.1 Caching Strategy
```python
class CacheManager:
    def __init__(self):
        self.redis = Redis()
        self.ttl = 300  # 5 minutes
    
    def cache_fragility_score(self, node_id: str, score: float):
        key = f"fragility:{node_id}"
        self.redis.setex(key, self.ttl, score)
    
    def get_cached_score(self, node_id: str) -> Optional[float]:
        key = f"fragility:{node_id}"
        return self.redis.get(key)
```

**Cache Invalidation Rules**:
- Invalidate on file modification
- Invalidate related nodes on dependency changes
- Periodic refresh every 5 minutes for active files

### 7.2 Graph Pruning
- Limit traversal depth to 5 hops
- Sample large neighborhoods (> 100 neighbors)
- Prune low-importance edges (< 10% coupling)

### 7.3 Batch Processing
- Group multiple small changes into batches
- Process batches every 500ms instead of per-keystroke
- Deduplicate redundant updates

### 7.4 Asynchronous Processing
- All ML inference runs asynchronously
- Non-blocking WebSocket communication
- Background workers for historical analysis

## 8. Error Handling

### 8.1 Graceful Degradation
- If GNN unavailable: Fall back to rule-based scoring
- If Neo4j unavailable: Use cached results
- If Redis unavailable: Direct database queries (slower)

### 8.2 Retry Logic
- Exponential backoff for failed API calls
- Maximum 3 retries for graph updates
- Circuit breaker pattern for external services

### 8.3 Error Reporting
```typescript
class ErrorReporter {
  reportError(error: Error, context: ErrorContext): void {
    // Log to console
    console.error(`[FragilityGraph] ${error.message}`, context);
    
    // Send to monitoring service
    this.monitoring.trackError(error, context);
    
    // Show user-friendly message
    this.ui.showNotification({
      type: 'error',
      message: this.getUserFriendlyMessage(error),
      actions: ['Retry', 'Report Bug']
    });
  }
}
```

## 9. Testing Strategy

### 9.1 Unit Tests
- All core algorithms (fragility calculation, pulse simulation)
- AST parsing logic
- Graph query functions
- 80%+ code coverage target

### 9.2 Integration Tests
- End-to-end flow from IDE to visualization
- Neo4j query performance tests
- WebSocket connection handling
- GNN inference accuracy tests

### 9.3 Performance Tests
- Load testing with 50 concurrent users
- Graph database scalability (100K+ nodes)
- Latency benchmarks (< 2s end-to-end)

### 9.4 Accuracy Tests
- Compare predictions with actual production failures
- Precision and recall metrics
- False positive rate monitoring

## 10. Deployment Architecture

### 10.1 Development Environment
```yaml
services:
  api:
    image: fragilityraph-api:latest
    ports: ["3000:3000"]
  
  ml-service:
    image: fragilityraph-ml:latest
    ports: ["8000:8000"]
    
  neo4j:
    image: neo4j:5.x
    ports: ["7687:7687", "7474:7474"]
    
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### 10.2 Production Architecture
- Kubernetes cluster with 3+ nodes
- Neo4j cluster (3 core servers, 2 read replicas)
- Redis Sentinel for high availability
- Load balancer for API servers
- Horizontal pod autoscaling based on CPU/memory

### 10.3 Monitoring
- Prometheus for metrics collection
- Grafana dashboards for visualization
- Alert rules for:
  - API latency > 2 seconds
  - Error rate > 1%
  - Memory usage > 80%
  - Neo4j query time > 1 second

## 11. Future Enhancements

### 11.1 Advanced ML Features
- Attention mechanisms to identify critical paths
- Reinforcement learning for optimal refactoring suggestions
- Transfer learning across different codebases

### 11.2 Extended Language Support
- Rust, C++, Swift, Kotlin
- Dynamic languages (Ruby, PHP)
- Domain-specific languages (SQL, GraphQL)

### 11.3 Architecture Improvements
- Distributed graph processing (Apache Spark GraphX)
- Streaming architecture (Apache Kafka)
- Event sourcing for audit trails

### 11.4 User Experience
- Voice commands for queries
- Natural language impact reports
- AR/VR visualization of code graphs
- Mobile app for monitoring