# FragilityGraph: The Winning Plan Summary

## 1. The Product
A real-time AI security & stability tool for developers.
- Primary Interface: VS Code Extension.
- Core Tech: Graph Neural Networks (GNN) + Amazon Bedrock.
- Value Prop: Predicts which code changes will break the system before they happen.

## 2. The "AWS-Native" Architecture (Budget: <$100)
- API: Amazon API Gateway (Free Tier - 1M requests).
- Compute: AWS Lambda (Serverless - pay only for what you use).
- AI Engine: PyTorch GNN running within Lambda + Amazon Bedrock (Claude 3 Haiku) for risk explanations.
- Database: Amazon DynamoDB (NoSQL used for Graph storage - cheap & scalable).
- Storage: Amazon S3 (Houses model weights and training data).
- Hosting: AWS Amplify (For the Web MVP Link).

## 3. The 4-Step Filter Strategy (Winning the Hackathon)
- Step 1 (PPT): Focus on "The Hidden Cost of Fragile Code." Show high-end AWS architecture diagrams.
- Step 2 (Video): Demonstrate REAL-TIME interaction. Show a code change in VS Code triggering an instant "Risk Warning" via the extension.
- Step 3 (MVP Link): Host a "Fragility Explorer" Web App using AWS Amplify. Judges can paste code snippets to see live GNN analysis without installing the extension.
- Step 4 (GitHub): Clean Python code using PyTorch Geometric and AWS SAM/CDK for infrastructure.

## 4. Key Selling Points for Judges
1. Privacy First: Code is parsed locally via Tree-sitter; only structural maps hit the cloud.
2. Hard Tech: Uses advanced Graph AI (GNNs), not just simple "wrappers."
3. Scalable & Native: Built 100% on AWS Serverless best practices.
4. Explanable AI: Uses Amazon Bedrock to explain complex technical risks in plain English.
