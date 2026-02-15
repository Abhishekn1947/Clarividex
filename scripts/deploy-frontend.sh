#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Deploy Frontend to S3 + CloudFront
# Usage: ./scripts/deploy-frontend.sh
#
# Required env vars (or fetched from terraform output):
#   NEXT_PUBLIC_API_URL  - Lambda Function URL
#   S3_BUCKET            - S3 bucket name
#   DISTRIBUTION_ID      - CloudFront distribution ID
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# Fetch values from terraform output if not set
if [[ -z "${S3_BUCKET:-}" || -z "${DISTRIBUTION_ID:-}" || -z "${NEXT_PUBLIC_API_URL:-}" ]]; then
  echo "=> Fetching config from terraform output..."
  S3_BUCKET="${S3_BUCKET:-$(cd "$TERRAFORM_DIR" && terraform output -raw frontend_s3_bucket)}"
  DISTRIBUTION_ID="${DISTRIBUTION_ID:-$(cd "$TERRAFORM_DIR" && terraform output -raw cloudfront_distribution_id)}"
  NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-$(cd "$TERRAFORM_DIR" && terraform output -raw function_url)}"
fi

echo "=> Config:"
echo "   S3_BUCKET:            $S3_BUCKET"
echo "   DISTRIBUTION_ID:      $DISTRIBUTION_ID"
echo "   NEXT_PUBLIC_API_URL:  $NEXT_PUBLIC_API_URL"

# Build frontend
echo ""
echo "=> Installing dependencies..."
cd "$FRONTEND_DIR"
npm ci

echo ""
echo "=> Building frontend (static export)..."
NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" npm run build

OUT_DIR="$FRONTEND_DIR/out"
if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: Build output directory not found at $OUT_DIR"
  exit 1
fi

# Upload to S3 - two-pass strategy for cache control
echo ""
echo "=> Uploading to S3..."

# Pass 1: HTML files - no cache (always revalidate)
echo "   Uploading HTML files (no-cache)..."
aws s3 sync "$OUT_DIR" "s3://$S3_BUCKET" \
  --exclude "*" \
  --include "*.html" \
  --cache-control "public,max-age=0,must-revalidate" \
  --delete

# Pass 2: Static assets (_next/static) - immutable, 1-year cache
echo "   Uploading static assets (immutable)..."
aws s3 sync "$OUT_DIR" "s3://$S3_BUCKET" \
  --exclude "*.html" \
  --cache-control "public,max-age=31536000,immutable" \
  --delete

# Invalidate CloudFront cache
echo ""
echo "=> Invalidating CloudFront cache..."
INVALIDATION_ID=$(aws cloudfront create-invalidation \
  --distribution-id "$DISTRIBUTION_ID" \
  --paths "/*" \
  --query 'Invalidation.Id' \
  --output text)

echo "   Invalidation ID: $INVALIDATION_ID"

# Get CloudFront URL
CF_URL=$(cd "$TERRAFORM_DIR" && terraform output -raw frontend_url)
echo ""
echo "=> Deploy complete!"
echo "   Frontend URL: $CF_URL"
echo ""
echo "   Note: CloudFront may take a few minutes to propagate."
