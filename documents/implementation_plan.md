# FragilityGraph Implementation Plan
## Budget-Conscious AWS Architecture ($100 Credit Limit)

## Executive Summary

FragilityGraph is a real-time code analysis system that predicts structural failures using Graph Neural Networks. This implementation plan is **optimized for a $100 AWS credit budget** while maintaining core functionality for hackathon demonstration and MVP validation.

**Key Constraint:** With only $100 in credits, we cannot use expensive managed services like Neptune ($350/month) or SageMaker inference endpoints ($100+/month). This plan leverages AWS free tiers aggressively and uses SageMaker **only for one-time model training**, not inference.

---

## 1. Budget-Optimized AWS Architecture

### 1.1 Complete Stack (Under $100)

```
┌─────────────────────────────────────────────┐
│        VS Code Extension (Local)            │
│  Tree-sitter parsing (client-side)          │
└─────────────────────────────────────────────┘
                    ↕ HTTPS
┌─────────────────────────────────────────────┐
│      API Gateway (FREE TIER)                │
│  1M requests/month included                 │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│      AWS Lambda (FREE TIER)                 │
│  - AST delta processor                      │
│  - Graph query handler                      │
│  - ML inference runner (in-process PyTorch) │
│  1M requests + 400K GB-seconds free         │
└─────────────────────────────────────────────┘
         ↕                       ↕
┌──────────────────┐    ┌────────────────────┐
│  DynamoDB        │    │  S3 (FREE TIER)    │
│  (FREE TIER)     │    │  - Repo snapshots  │
│  - Graph nodes   │    │  - Model weights   │
│  - Graph edges   │    │  5GB free          │
│  25GB + 25 WCU   │    └────────────────────┘
└──────────────────┘
         ↕
┌─────────────────────────────────────────────┐
│   SageMaker (ONE-TIME TRAINING ONLY)        │
│   ml.m5.large × 2 hours = ~$0.30            │
│   OR ml.p3.2xlarge × 2 hours = ~$8.50       │
│   Export weights → S3 → Load in Lambda      │
└─────────────────────────────────────────────┘
```

### 1.2 Why This Architecture?

**Cost Breakdown:**
| Service | Usage | Monthly Cost | Notes |
|---------|-------|--------------|-------|
| API Gateway | 10K requests | **$0** | Free tier: 1M req/month |
| Lambda | 100K invocations | **$0** | Free tier: 1M req + 400K GB-sec |
| DynamoDB | 1GB data, 1M reads | **$0-2** | Free tier: 25GB + 25 RCU/WCU |
| S3 | 5GB storage | **$0** | Free tier: 5GB |
| SageMaker Training | 2 hours once | **$0.30-8.50** | One-time cost |
| CloudWatch | Basic logs | **$0** | Free tier: 5GB logs |
| **TOTAL** | | **$2-15** | **$85-98 buffer remaining** |

**Key Decisions:**

1. ❌ **No SageMaker Inference Endpoints** - Would cost $100+/month
   - ✅ Instead: Train model once, export weights to S3, load in Lambda
   - Trade-off: Slightly slower inference (~500ms vs 200ms) but $100/month savings

2. ❌ **No Amazon Neptune** - Would cost $350+/month
   - ✅ Instead: DynamoDB with denormalized graph structure
   - Trade-off: More complex queries but completely free (within 25GB)

3. ❌ **No Amazon Bedrock** - Would cost $15 per 1M tokens
   - ✅ Instead: Template-based explanations
   - Trade-off: Less natural language but saves $50-200/month

4. ❌ **No ElastiCache Redis** - Would cost $50+/month
   - ✅ Instead: In-memory caching within Lambda (limited but free)
   - Trade-off: Cache doesn't persist across invocations

5. ❌ **No ECS/Fargate** - Would cost $50+/month
   - ✅ Instead: Lambda handles all API logic
   - Trade-off: 15-minute timeout (sufficient for our use case)

---

## 2. Service-by-Service Implementation

### 2.1 AWS Lambda (Primary Compute - FREE)

**Why Lambda:**
- **Free Tier:** 1M requests/month + 400,000 GB-seconds
- **Cost After Free Tier:** $0.20 per 1M requests (incredibly cheap)
- **Perfect for:** Bursty workloads like code analysis
- **Your testing needs:** Even 100K requests in testing = $0

**Lambda Functions:**

#### **Function 1: ast-delta-processor**
```python
# Runtime: Python 3.11
# Memory: 512MB
# Timeout: 30 seconds
# Trigger: API Gateway POST /analyze

import json
import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
nodes_table = dynamodb.Table('fragilityraph-nodes')
edges_table = dynamodb.Table('fragilityraph-edges')

def lambda_handler(event, context):
    # Parse incoming AST delta from IDE
    body = json.loads(event['body'])
    repo_id = body['repo_id']
    changed_file = body['file']
    added_nodes = body['added_nodes']  # Functions/classes added
    removed_nodes = body['removed_nodes']
    
    # Update DynamoDB graph
    for node in added_nodes:
        nodes_table.put_item(Item={
            'repo_id': repo_id,
            'node_id': f"{changed_file}::{node['name']}",
            'type': node['type'],  # function, class, module
            'file': changed_file,
            'complexity': node.get('complexity', 1),
            'line_start': node['line_start'],
            'line_end': node['line_end'],
            'timestamp': datetime.now().isoformat()
        })
    
    for node_id in removed_nodes:
        nodes_table.delete_item(Key={'node_id': node_id})
    
    return {
        'statusCode': 200,
        'body': json.dumps({'status': 'graph_updated'})
    }
```

#### **Function 2: ml-inference-runner**
```python
# Runtime: Python 3.11
# Memory: 3GB (for PyTorch model)
# Timeout: 60 seconds
# Trigger: API Gateway POST /predict

import json
import boto3
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

# Download model weights from S3 on cold start (cached for warm invocations)
s3 = boto3.client('s3')
MODEL_BUCKET = 'fragilityraph-models'
MODEL_KEY = 'graphsage_v1.pth'

# Global variable (persists across warm Lambda invocations)
model = None

def load_model():
    global model
    if model is None:
        # Download from S3
        s3.download_file(MODEL_BUCKET, MODEL_KEY, '/tmp/model.pth')
        
        # Load PyTorch model
        model = FragilityGNN(input_dim=64, hidden_dim=128)
        model.load_state_dict(torch.load('/tmp/model.pth'))
        model.eval()
    return model

class FragilityGNN(torch.nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.sigmoid(self.conv3(x, edge_index))
        return x

def lambda_handler(event, context):
    # Load model (cached after first invocation)
    model = load_model()
    
    # Parse request
    body = json.loads(event['body'])
    node_features = torch.tensor(body['node_features'])  # [num_nodes, 64]
    edge_index = torch.tensor(body['edge_index'])  # [2, num_edges]
    
    # Run inference
    with torch.no_grad():
        data = Data(x=node_features, edge_index=edge_index)
        predictions = model(data.x, data.edge_index)
    
    # Convert to fragility scores (0-100)
    fragility_scores = (predictions.squeeze() * 100).tolist()
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'fragility_scores': fragility_scores,
            'inference_time_ms': context.get_remaining_time_in_millis()
        })
    }
```

**Lambda Cost Calculation:**
```
Your testing phase (1 month):
- 100,000 API calls
- Average 1GB memory × 2 seconds per call = 200,000 GB-seconds
- Cost: $0 (well within 1M requests + 400K GB-seconds free tier)

Even heavy testing:
- 500,000 API calls  
- 1,000,000 GB-seconds
- Cost: Still $0 (barely exceeds free tier)
- Worst case overage: ~$10
```

### 2.2 Amazon DynamoDB (Graph Storage - FREE)

**Why DynamoDB:**
- **Free Tier:** 25GB storage + 25 RCU + 25 WCU
- **Perfect for:** Small-to-medium graphs (up to 100K nodes)
- **Your needs:** Sample repo with 10K nodes = ~50MB = **$0**

**Table Schema:**

**Table 1: fragilityraph-nodes**
```javascript
{
  "node_id": "auth/login.py::authenticate_user",  // Partition key
  "repo_id": "flask-demo",  // GSI partition key
  "type": "function",  // function, class, module
  "file": "auth/login.py",
  "name": "authenticate_user",
  "complexity": 12,
  "loc": 45,
  "fragility_score": 0.82,  // Updated by ML model
  "callers": ["auth/login.py::login_handler", "api/v1.py::verify"],  // Denormalized
  "callees": ["db/users.py::get_user", "auth/tokens.py::generate_token"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Table 2: fragilityraph-edges**
```javascript
{
  "edge_id": "auth/login.py::login_handler->authenticate_user",  // Partition key
  "source_node": "auth/login.py::login_handler",
  "target_node": "auth/login.py::authenticate_user",
  "edge_type": "CALLS",  // CALLS, IMPORTS, CO_CHANGES_WITH
  "frequency": 1250,  // Number of times called
  "weight": 0.8  // Importance weight
}
```

**Query Patterns:**
```python
# Get all dependencies of a node (BFS traversal)
def get_dependencies(node_id, depth=3):
    visited = set()
    queue = [(node_id, 0)]
    dependencies = []
    
    while queue:
        current_id, current_depth = queue.pop(0)
        if current_depth > depth or current_id in visited:
            continue
        
        visited.add(current_id)
        
        # Query node
        node = nodes_table.get_item(Key={'node_id': current_id})['Item']
        dependencies.append(node)
        
        # Add callees to queue
        for callee in node.get('callees', []):
            queue.append((callee, current_depth + 1))
    
    return dependencies

# Find high-fragility nodes
response = nodes_table.scan(
    FilterExpression='fragility_score > :threshold',
    ExpressionAttributeValues={':threshold': 0.7}
)
```

**DynamoDB Cost Calculation:**
```
Your graph (10K nodes + 50K edges):
- Storage: ~100MB (well within 25GB free tier)
- Reads: 10K/day = 300K/month (within 25 RCU free tier)
- Writes: 1K/day = 30K/month (within 25 WCU free tier)
- Cost: $0

Even with 100K nodes:
- Storage: ~1GB (still free)
- Cost: $0-2/month maximum
```

### 2.3 Amazon S3 (Model & Repo Storage - FREE)

**Why S3:**
- **Free Tier:** 5GB storage, 20K GET requests, 2K PUT requests
- **Your needs:** Model weights (50MB) + repo snapshots (500MB) = **$0**

**Bucket Structure:**
```
fragilityraph-data/
├── models/
│   ├── graphsage_v1.pth          (PyTorch weights, 50MB)
│   ├── training_history.json      (metrics, 1MB)
│   └── feature_scaler.pkl         (preprocessing, 1MB)
├── repos/
│   ├── flask-demo.tar.gz          (200MB compressed)
│   └── express-js.tar.gz          (150MB compressed)
└── training-data/
    ├── failure_labels.csv         (historical failures, 10MB)
    └── feature_matrix.npz         (training features, 50MB)
```

**S3 Cost Calculation:**
```
Your usage:
- Storage: 500MB (within 5GB free tier)
- Lambda downloads model on cold start: ~50 requests/day
- Cost: $0
```

### 2.4 Amazon SageMaker (ONE-TIME Training - $5-10)

**Critical: Training Only, NOT Inference**

**Why:**
- Training a small GNN takes 1-2 hours on ml.m5.large ($0.30 total)
- Export model weights to S3
- Lambda loads weights for inference (no ongoing cost)
- **Saves $100+/month** vs. SageMaker inference endpoint

**Training Job Configuration:**
```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Use cheapest instance
estimator = PyTorch(
    entry_point='train.py',
    role=sagemaker_role,
    instance_type='ml.m5.large',  # $0.134/hour (cheapest)
    instance_count=1,
    framework_version='2.1.0',
    py_version='py311',
    hyperparameters={
        'epochs': 100,
        'batch-size': 64,
        'learning-rate': 0.001,
        'hidden-dim': 128
    }
)

# Train once
estimator.fit({'training': 's3://fragilityraph-data/training-data/'})

# Export model weights to S3 (happens automatically)
# No need to deploy inference endpoint!
```

**Training Script (train.py):**
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import argparse
import os

def train():
    # Parse hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    args = parser.parse_args()
    
    # Load training data from /opt/ml/input/data/training/
    train_data = load_graph_data()
    
    # Initialize model
    model = FragilityGNN(input_dim=64, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = F.binary_cross_entropy(out, train_data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Save model to /opt/ml/model/ (automatically uploaded to S3)
    model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'graphsage_v1.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    train()
```

**SageMaker Cost Calculation:**
```
Option 1 (CPU only):
- ml.m5.large: $0.134/hour
- Training time: 2 hours
- Total: $0.27 one-time

Option 2 (GPU for faster training):
- ml.p3.2xlarge: $4.284/hour
- Training time: 30 minutes
- Total: $2.14 one-time

Budget impact: $0.30-8.50 depending on instance choice
```

### 2.5 API Gateway (API Endpoints - FREE)

**Why API Gateway:**
- **Free Tier:** 1M API calls/month
- **Your testing:** 10K-100K calls = **$0**
- **After free tier:** $3.50 per 1M calls (extremely cheap)

**Endpoints:**
```
POST   /analyze        → ast-delta-processor Lambda
POST   /predict        → ml-inference-runner Lambda  
GET    /fragility/{file} → Query DynamoDB directly via Lambda
GET    /heatmap/{repo}   → Aggregate DynamoDB scan via Lambda
```

**API Gateway Configuration:**
```yaml
Resources:
  FragilityAPI:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: fragilityraph-api
      
  AnalyzeResource:
    Type: AWS::ApiGateway::Resource
    Properties:
      ParentId: !GetAtt FragilityAPI.RootResourceId
      PathPart: analyze
      
  AnalyzeMethod:
    Type: AWS::ApiGateway::Method
    Properties:
      HttpMethod: POST
      ResourceId: !Ref AnalyzeResource
      RestApiId: !Ref FragilityAPI
      AuthorizationType: AWS_IAM  # Free, secure
      Integration:
        Type: AWS_PROXY
        IntegrationHttpMethod: POST
        Uri: !Sub arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${ASTDeltaProcessor.Arn}/invocations
```

### 2.6 CloudWatch (Monitoring - FREE)

**Free Tier:**
- 5GB log ingestion
- 10 custom metrics
- 10 alarms

**Your usage:**
- Lambda logs: ~100MB/month
- Basic metrics: API latency, error rate
- **Cost: $0**

---

## 3. Development Roadmap

### Week 1: Infrastructure Setup & Lambda Development

**Day 1-2: AWS Setup**
- ✅ Create DynamoDB tables (nodes, edges)
- ✅ Create S3 bucket for models
- ✅ Set up IAM roles for Lambda
- ✅ Deploy API Gateway

**Day 3-4: Lambda Functions**
- ✅ Write `ast-delta-processor` Lambda
- ✅ Write `ml-inference-runner` Lambda with PyTorch
- ✅ Test locally with SAM CLI
- ✅ Deploy to AWS

**Day 5-7: DynamoDB Integration**
- ✅ Implement graph ingestion from sample repo
- ✅ Write query functions for dependency traversal
- ✅ Populate with Flask repo (10K nodes)
- ✅ Benchmark query performance

### Week 2: ML Model Training & Integration

**Day 8-10: Feature Engineering**
- ✅ Extract node features (complexity, degree, LOC)
- ✅ Create training dataset with synthetic labels
- ✅ Store in S3

**Day 11-12: SageMaker Training**
- ✅ Write training script (train.py)
- ✅ Run SageMaker training job (ml.m5.large, 2 hours, $0.30)
- ✅ Export model weights to S3

**Day 13-14: Lambda Inference Integration**
- ✅ Update Lambda to load model from S3
- ✅ Test inference performance (target: < 1 second)
- ✅ Implement in-memory caching for warm invocations

### Week 3-4: VS Code Extension

**Day 15-18: Extension Development**
- ✅ Scaffold VS Code extension
- ✅ Integrate Tree-sitter for local AST parsing
- ✅ Send deltas to API Gateway
- ✅ Receive fragility scores

**Day 19-22: Visualization**
- ✅ Render heatmap in sidebar
- ✅ Show blast radius graph
- ✅ Display inline risk indicators

**Day 23-28: Polish & Testing**
- ✅ End-to-end testing (IDE → Lambda → DynamoDB → Response)
- ✅ Optimize for latency (target: < 2 seconds)
- ✅ Handle edge cases
- ✅ Write documentation

---

## 4. 24-Hour Proof of Concept Plan

### Hour 0-6: AWS Setup
1. Create DynamoDB tables via CloudFormation
2. Create S3 bucket
3. Deploy Lambda functions with placeholder code
4. Set up API Gateway routes
5. Test "Hello World" end-to-end

### Hour 6-12: Graph Ingestion
1. Write Python script to parse Flask repo with Tree-sitter
2. Extract 1000+ nodes (functions/classes)
3. Upload to DynamoDB
4. Verify with DynamoDB queries

### Hour 12-18: ML Training
1. Create simple training dataset (50 nodes with synthetic labels)
2. Write minimal train.py script
3. Launch SageMaker training job (ml.m5.large, 1 hour, $0.15)
4. Download model weights from S3

### Hour 18-22: Lambda Inference
1. Update Lambda to load model
2. Implement feature extraction
3. Run inference on sample code change
4. Return fragility scores

### Hour 22-24: API Testing & Demo
1. Test API with curl/Postman
2. Verify latency < 2 seconds
3. Prepare demo showing:
   - Code change input
   - Fragility scores output
   - Affected components list

**Success Criteria:**
- API responds in < 2 seconds
- Fragility scores between 0-100
- Can process 10 code changes without errors
- **Total cost: < $5**

---

## 5. Cost Breakdown & Budget Safety

### Expected Costs (1 Month Testing Phase)

| Service | Usage | Cost | Notes |
|---------|-------|------|-------|
| **Lambda** | 100K invocations, 500K GB-sec | **$0** | Within free tier |
| **API Gateway** | 50K requests | **$0** | Within free tier |
| **DynamoDB** | 1GB storage, 500K reads | **$0** | Within free tier |
| **S3** | 500MB storage, 10K requests | **$0** | Within free tier |
| **SageMaker Training** | ml.m5.large × 2 hours | **$0.30** | One-time |
| **CloudWatch** | 100MB logs | **$0** | Within free tier |
| **Data Transfer** | 10GB out | **$1** | Minimal |
| **Buffer** | Unexpected overages | **$5** | Safety margin |
| **TOTAL** | | **~$6.30** | **$93.70 remaining** |

### Worst Case Scenario

Even if you exceed free tiers:
- Lambda: 1M invocations = $0.20
- DynamoDB: 10GB storage = $2.50
- S3: 10GB storage = $0.23
- API Gateway: 1M requests = $3.50
- **Total worst case: $15** (still $85 under budget)

### Cost Monitoring

**Set up billing alerts:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name fragilityraph-budget-alert \
  --alarm-description "Alert if spending > $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:billing-alerts
```

---

## 6. What We're Sacrificing (And Why It's Okay for MVP)

### ❌ Things We Removed from Original Plan

1. **Amazon Neptune** ($350/month)
   - **Alternative:** DynamoDB with denormalized graph
   - **Trade-off:** Slower graph queries (200ms vs 50ms)
   - **Impact:** Acceptable for MVP, can migrate later

2. **SageMaker Inference Endpoint** ($100/month)
   - **Alternative:** Load model in Lambda
   - **Trade-off:** Cold start latency (2s first call, then 500ms)
   - **Impact:** Minor UX issue, saves $100/month

3. **Amazon Bedrock** ($50-200/month)
   - **Alternative:** Template-based explanations
   - **Trade-off:** Less natural language
   - **Impact:** Can add in Phase 2 with real budget

4. **ElastiCache Redis** ($50/month)
   - **Alternative:** In-memory Lambda caching
   - **Trade-off:** Cache doesn't persist across Lambda cold starts
   - **Impact:** Slightly higher latency, but manageable

5. **ECS/Fargate** ($50/month)
   - **Alternative:** Lambda for all compute
   - **Trade-off:** 15-minute timeout limit
   - **Impact:** No issue for our use case (queries finish in < 30 seconds)

### ✅ What We Keep (Core Functionality)

- ✅ Real-time AST parsing in IDE
- ✅ Graph dependency tracking
- ✅ GNN-based fragility prediction
- ✅ Blast radius calculation
- ✅ Sub-2-second response time
- ✅ Support for 10K+ node codebases
- ✅ Visual heatmaps and risk indicators

**Bottom line:** We're delivering 90% of the core value at 5% of the cost.

---

## 7. Scaling Path (When You Get Real Budget)

### Phase 1: MVP (Current - $100 budget)
- Lambda + DynamoDB + S3
- 100 concurrent developers
- 10K node codebases

### Phase 2: Startup ($500/month budget)
- Add ElastiCache for persistent caching
- Upgrade to SageMaker inference endpoint
- Support 500 concurrent developers
- 100K node codebases

### Phase 3: Growth ($2K/month budget)
- Migrate DynamoDB → Neptune for faster graph queries
- Add Bedrock for natural language explanations
- Multi-region deployment
- 5K concurrent developers

### Phase 4: Enterprise ($10K+/month budget)
- Full managed service suite
- Custom model training per organization
- Dedicated infrastructure
- 50K+ concurrent developers

---

## 8. Success Metrics

### Technical Metrics
- ✅ API latency p95: < 2 seconds
- ✅ Lambda cold start: < 3 seconds
- ✅ Lambda warm invocation: < 500ms
- ✅ DynamoDB query time: < 200ms
- ✅ Model inference time: < 1 second
- ✅ End-to-end latency: < 2.5 seconds

### Cost Metrics
- ✅ Monthly AWS spend: < $20
- ✅ Cost per prediction: < $0.0001
- ✅ Budget utilization: < 20% of $100 credit

### Functional Metrics
- ✅ Successfully analyze 10K node repository
- ✅ Detect dependencies 5 hops deep
- ✅ Calculate fragility scores for all functions
- ✅ Generate blast radius for code changes
- ✅ Handle 100 concurrent API requests

---

## 9. FAQ: Addressing Concerns

**Q: Can Lambda really handle ML inference?**
A: Yes! Small GNN models (< 100MB) work great in Lambda with 3GB memory. Cold starts are ~2-3 seconds, warm invocations are ~500ms.

**Q: What about Lambda's 15-minute timeout?**
A: Our queries finish in < 30 seconds. Lambda timeout is not an issue.

**Q: Will DynamoDB scale to large codebases?**
A: For MVP (10K-100K nodes), yes. For enterprise scale (1M+ nodes), migrate to Neptune in Phase 2.

**Q: What if I exceed the free tier?**
A: Worst case: $15/month. Still well under $100 budget. Set billing alarms at $50 to be safe.

**Q: Can I really test thoroughly with $100?**
A: Absolutely. Free tiers cover 1M Lambda calls + 1M API requests. Even aggressive testing stays under $20.

---

## 10. Conclusion

This implementation plan delivers a **fully functional FragilityGraph MVP** while staying **well under the $100 AWS credit budget**.

**Key Achievements:**
- ✅ Real-time code analysis with GNN predictions
- ✅ Sub-2-second response time
- ✅ Support for 10K+ node repositories
- ✅ Total cost: $5-15/month (85-95% under budget)
- ✅ Room for extensive testing and iteration

**Trade-offs Accepted:**
- Slightly slower than ideal (2s vs 500ms) due to Lambda cold starts
- DynamoDB queries slower than Neptune (but acceptable)
- No natural language explanations (can add later with Bedrock)

**This plan is realistic, achievable, and demonstrates core value without breaking the bank.**
