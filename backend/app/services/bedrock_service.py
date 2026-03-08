"""
Bedrock AI Explanation Service
Calls Amazon Bedrock (Nova Lite) to generate explanations,
file summaries, and change impact analysis.
"""
import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


def _get_bedrock_client():
    kwargs = {"region_name": settings.AWS_REGION}
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN
    return boto3.client("bedrock-runtime", **kwargs)


def _invoke_bedrock(prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
    """Send a prompt to Bedrock and return the text response."""
    try:
        client = _get_bedrock_client()
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
            }
        )
        response = client.invoke_model(
            modelId=settings.BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        text = result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        return text.strip() if text else ""
    except ClientError as e:
        logger.warning("Bedrock API error: %s", e)
    except Exception as e:
        logger.warning("Bedrock call failed: %s", e)
    return ""


# ── 1. Node-level fragility explanation ────────────────────
def explain_fragility(
    function_name: str,
    code_snippet: str,
    fragility_score: float,
    dependencies: list[str] | None = None,
) -> str:
    """Generate a short AI explanation for why a function is fragile."""
    deps_text = ", ".join(dependencies) if dependencies else "none identified"

    prompt = (
        f"You are a senior software engineer. In 2-3 concise sentences, explain why "
        f"the function `{function_name}` (fragility score {fragility_score:.0f}/100) "
        f"is risky to modify. Its dependencies are: {deps_text}.\n"
        f"Code:\n```\n{code_snippet[:500]}\n```"
    )

    text = _invoke_bedrock(prompt, max_tokens=200)
    if text:
        return text

    # Fallback
    risk = "high" if fragility_score >= 70 else "moderate" if fragility_score >= 40 else "low"
    return (
        f"`{function_name}` has {risk} fragility ({fragility_score:.0f}/100) "
        f"with dependencies on {deps_text}. Changes here may propagate failures."
    )


# ── 2. File-level summary ─────────────────────────────────
def summarize_file(file_path: str, content: str, node_count: int, edge_count: int, max_fragility: float) -> str:
    """Generate a 2-3 sentence summary of a file's purpose and risk posture."""
    filename = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]
    prompt = (
        f"You are a senior code reviewer. In 2-3 sentences, summarize the purpose of "
        f"the Python file `{filename}`. It has {node_count} functions/classes, "
        f"{edge_count} internal dependencies, and a max fragility score of "
        f"{max_fragility:.0f}/100. Focus on what the file does and its risk posture.\n"
        f"Code (first 800 chars):\n```\n{content[:800]}\n```"
    )

    text = _invoke_bedrock(prompt, max_tokens=200)
    if text:
        return text

    # Fallback
    risk = "high-risk" if max_fragility >= 70 else "moderate-risk" if max_fragility >= 40 else "low-risk"
    return (
        f"`{filename}` contains {node_count} code elements with {edge_count} dependencies. "
        f"Max fragility: {max_fragility:.0f}/100 ({risk}). "
        f"Review tightly-coupled functions before making changes."
    )


# ── 3. Change impact analysis ─────────────────────────────
def analyze_change_impact(file_path: str, content: str, change_description: str, function_names: list[str]) -> list[str]:
    """
    Given a file and a user-described change, return a list of function names
    likely affected by that change.
    """
    fn_list = ", ".join(function_names) if function_names else "none"
    prompt = (
        f"You are a code impact analysis expert. A developer wants to make this change "
        f"to the file `{file_path.split('/')[-1]}`:\n\n"
        f"Change: \"{change_description}\"\n\n"
        f"The file contains these functions/classes: {fn_list}\n\n"
        f"Code (first 1200 chars):\n```\n{content[:1200]}\n```\n\n"
        f"Return ONLY a JSON array of function/class names that would be affected by this change. "
        f"Example: [\"func_a\", \"ClassB\"]. Return [] if no functions are affected. "
        f"Do NOT include any explanation, just the JSON array."
    )

    text = _invoke_bedrock(prompt, max_tokens=200, temperature=0.1)
    if text:
        try:
            # Try to extract JSON array from response
            # Handle cases where model wraps in markdown code block
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(cleaned)
            if isinstance(result, list):
                return [str(x) for x in result]
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse impact analysis response: %s", text)
            # Try to extract function names from text
            affected = [fn for fn in function_names if fn.lower() in text.lower()]
            if affected:
                return affected

    # Fallback: simple keyword matching
    affected = []
    change_lower = change_description.lower()
    for fn in function_names:
        if fn.lower() in change_lower:
            affected.append(fn)
    return affected if affected else function_names[:3]  # Return top 3 as fallback
