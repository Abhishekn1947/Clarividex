#!/usr/bin/env bash
# =============================================================================
# Clarividex - Build and Push Lambda Container Image to ECR
#
# Usage:
#   ./docker/build-and-push.sh                    # uses defaults
#   ./docker/build-and-push.sh us-east-1 clarividex
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker running
#   - ECR repository exists (created by Terraform)
# =============================================================================

set -euo pipefail

# Configuration
AWS_REGION="${1:-us-east-1}"
APP_NAME="${2:-clarividex}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
LAMBDA_FUNCTION="${APP_NAME}-prod"

echo "============================================="
echo "  Clarividex - Build & Deploy to AWS Lambda"
echo "============================================="
echo "  Region:    ${AWS_REGION}"
echo "  ECR Repo:  ${ECR_REPO}"
echo "  Tag:       ${COMMIT_SHA}"
echo "  Function:  ${LAMBDA_FUNCTION}"
echo "============================================="

# Step 1: Login to ECR
echo "[1/5] Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Step 2: Build Docker image
echo "[2/5] Building Docker image..."
docker build \
  -f docker/Dockerfile.lambda \
  -t "${APP_NAME}:${COMMIT_SHA}" \
  -t "${APP_NAME}:latest" \
  .

# Step 3: Tag for ECR
echo "[3/5] Tagging image for ECR..."
docker tag "${APP_NAME}:${COMMIT_SHA}" "${ECR_REPO}:${COMMIT_SHA}"
docker tag "${APP_NAME}:latest" "${ECR_REPO}:latest"

# Step 4: Push to ECR
echo "[4/5] Pushing to ECR..."
docker push "${ECR_REPO}:${COMMIT_SHA}"
docker push "${ECR_REPO}:latest"

# Step 5: Update Lambda function
echo "[5/5] Updating Lambda function..."
aws lambda update-function-code \
  --function-name "${LAMBDA_FUNCTION}" \
  --image-uri "${ECR_REPO}:${COMMIT_SHA}" \
  --region "${AWS_REGION}" \
  --no-cli-pager

# Wait for update to complete
echo "Waiting for Lambda update to complete..."
aws lambda wait function-updated \
  --function-name "${LAMBDA_FUNCTION}" \
  --region "${AWS_REGION}"

echo ""
echo "============================================="
echo "  Deploy complete!"
echo "  Image: ${ECR_REPO}:${COMMIT_SHA}"
echo "============================================="

# Get the function URL and run smoke test
FUNCTION_URL=$(aws lambda get-function-url-config \
  --function-name "${LAMBDA_FUNCTION}" \
  --region "${AWS_REGION}" \
  --query 'FunctionUrl' \
  --output text 2>/dev/null || echo "")

if [ -n "${FUNCTION_URL}" ]; then
  echo ""
  echo "Running smoke test..."
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${FUNCTION_URL}api/v1/health" --max-time 30)
  if [ "${HTTP_CODE}" = "200" ]; then
    echo "  Health check: PASSED (HTTP ${HTTP_CODE})"
  else
    echo "  Health check: FAILED (HTTP ${HTTP_CODE})"
    echo "  URL: ${FUNCTION_URL}api/v1/health"
  fi
  echo ""
  echo "  API URL: ${FUNCTION_URL}"
  echo "  Health:  ${FUNCTION_URL}api/v1/health"
fi
