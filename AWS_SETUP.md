# AWS Setup Instructions for FragilityGraph

This document provides step-by-step instructions to set up the required AWS resources for the FragilityGraph project while staying within a budget-conscious hackathon credit limit.

## 1. Prerequisites
- AWS Account with Console access.
- AWS CLI configured locally (optional, but helpful).

## 2. Resource Setup

### A. Amazon S3 (Model Storage)
1. Go to **S3 Console**.
2. Click **Create bucket**.
3. **Bucket name:** `fragility-graph-models-[your-unique-id]`.
4. **Region:** Choose `us-east-1` (typical for credits).
5. Keep other settings default and click **Create bucket**.
6. Copy the bucket name into your `.env` as `AWS_S3_BUCKET_NAME`.

### B. Amazon Bedrock (AI Explanations)
1. Go to **Amazon Bedrock Console**.
2. **Model access:** In the left sidebar, click **Model access**.
3. Click **Edit**, then request access to **Amazon -> Nova Lite** (and **Nova Micro**).
   - *Note: Amazon-native models like Nova usually DO NOT require a credit card on file, unlike Anthropic/Meta models.*
4. Once access is granted (usually instant), note the region.
5. Set `AWS_REGION` in your `.env`.

### C. Credential Management 
**You have two options for managing credentials:**

1. **Option 1: AWS CLI (Recommended for Local Dev)**
   - Run `aws configure` in your terminal.
   - Enter your Access Key, Secret Key, and Region.
   - Boto3 will automatically pick these up from `~/.aws/credentials`.
   - **Leave `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` blank/commented in your `.env`.**

2. **Option 2: `.env` file (Required for Docker)**
   - Copy the **Access Key ID** and **Secret Access Key** into your `.env`.
   - **WARNING:** Never commit your `.env` file to GitHub. It is already added to `.gitignore`.

## 3. Environment Variables Summary
Update your `.env` file with the following:
- `AWS_REGION`: [e.g., us-east-1]
- `AWS_S3_BUCKET_NAME`: [From S3 setup]
- `BEDROCK_MODEL_ID`: `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- `AWS_ACCESS_KEY_ID`: (Only if using Option 2)
- `AWS_SECRET_ACCESS_KEY`: (Only if using Option 2)

## 4. Cost Management
- **S3:** Charging is based on storage; 1GB is negligible (<$0.10/mo).
- **Bedrock (Claude 3.5 Haiku):** 
  - Input: $0.80 per 1M tokens.
  - Output: $4.00 per 1M tokens.
  - *Extremely cheap for experimental use (fractions of a cent per demo call).*
- **Monitoring:** Check the **AWS Billing Dashboard** regularly.

### D. Setting up a Budget Alarm (Crucial!)
To ensure you don't accidentally blow through your $300, set an alarm:
1. Search for **Budgets** in the AWS Console.
2. Click **Create budget** -> **Cost budget (Recommended)**.
3. **Period:** Monthly.
4. **Budgeted amount:** $50 (or whatever you're comfortable with).
5. **Alerts:** Add an email alert when you reach 80% of this budget.
   - This prevents a "runaway loop" in your code from costing you all your credits in one night.
