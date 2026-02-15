# AWS Deployment — Clarividex (Terraform)

## Architecture (Deployed)

```
Browser
  |-- Static assets --> CloudFront (PriceClass_100) --> S3 (private, OAC)
  |-- API calls ------> Lambda Function URL (direct HTTPS, CORS via FastAPI)
```

- **Frontend**: Next.js 16 static export on S3 + CloudFront
- **Backend**: FastAPI in Docker on Lambda (ECR image, Mangum adapter)
- **No EC2, no ALB, no RDS** — fully serverless, ~$0-2/month at low usage

---

## Live URLs

| Resource | URL |
|----------|-----|
| Frontend | `https://dy9y276gfap4k.cloudfront.net` |
| API | `https://iqi7i4npsgrpf4otdpraqguefy0usjxv.lambda-url.us-east-1.on.aws` |
| Health Check | `https://iqi7i4npsgrpf4otdpraqguefy0usjxv.lambda-url.us-east-1.on.aws/api/v1/health` |

---

## Monthly Cost Estimate

| Resource | Free Tier | Post-Free Tier |
|----------|-----------|---------------|
| Lambda (1GB, ~45s/invoke) | $0 | $0.50-5.00 |
| ECR (Docker images) | $0 | $0.10 |
| S3 (~5MB frontend) | $0 | $0.01 |
| CloudFront (PriceClass_100) | $0 | $0.50-1.00 |
| CloudWatch Logs | $0 | $0.50 |
| SSM Parameter Store | $0 | $0 |
| **AWS Total** | **~$0** | **~$2-7** |
| Gemini API (~200 preds/mo) | $0 (free tier) | ~$1-3 |
| **Grand Total** | **~$0** | **~$3-10** |

### Cost Per Prediction

| Operation | Cost |
|-----------|------|
| Gemini 2.0 Flash — input (~2K tokens) | ~$0.0002 |
| Gemini 2.0 Flash — output (~4K tokens) | ~$0.004 |
| Lambda compute (1GB x 45s) | ~$0.0007 |
| CloudFront + S3 | negligible |
| **Total per prediction** | **~$0.005** |

---

## Terraform Modules

```
terraform/
  main.tf                        # Root: provider, module wiring
  variables.tf                   # Input variables (region, app_name, secrets)
  outputs.tf                     # URLs, bucket names, distribution IDs
  terraform.tfvars               # Actual values (GITIGNORED - never committed)
  modules/
    ecr/                         # Container registry for Lambda image
    secrets/                     # SSM Parameter Store (API keys)
    lambda/                      # Lambda function + Function URL + IAM
    frontend/                    # S3 bucket + CloudFront + OAC + SPA rewrite
    monitoring/                  # CloudWatch alarms + SNS email alerts
    warmup/                      # EventBridge rule (ping every 5min)
```

**~27 Terraform resources total.**

---

## Deployment Workflow

### Initial Deploy

```bash
# 1. Configure secrets
cd terraform
cp terraform.tfvars.example terraform.tfvars  # fill in API keys & AWS creds

# 2. Deploy infrastructure
terraform init && terraform apply
# Creates: ECR, Lambda, S3, CloudFront, monitoring, warmup

# 3. Build & push backend Docker image
# (automated via GitHub Actions on push to main, or manually):
aws ecr get-login-password | docker login --username AWS --password-stdin <ECR_URL>
docker build -f docker/Dockerfile.lambda -t clarividex .
docker tag clarividex:latest <ECR_URL>:latest
docker push <ECR_URL>:latest
aws lambda update-function-code --function-name clarividex-prod --image-uri <ECR_URL>:latest

# 4. Deploy frontend
./scripts/deploy-frontend.sh
# Builds static export, syncs to S3, invalidates CloudFront
```

### Ongoing Updates

- **Backend**: Push to `main` branch triggers GitHub Actions (build + push + Lambda update)
- **Frontend**: Push to `main` with `frontend/**` changes triggers GitHub Actions (build + S3 sync + CF invalidation)
- **Infrastructure**: `terraform apply` for any Terraform changes

---

## CI/CD Pipelines (GitHub Actions)

| Workflow | Trigger | What It Does |
|----------|---------|-------------|
| `ci.yml` | PR to main | Lint + build + test (frontend & backend) |
| `deploy.yml` | Push to main (`backend/**`) | Build Docker, push ECR, update Lambda |
| `deploy-frontend.yml` | Push to main (`frontend/**`) | Build static export, sync S3, invalidate CF |

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `APP_SECRET_KEY` | Application secret |
| `LAMBDA_FUNCTION_URL` | Lambda Function URL (for NEXT_PUBLIC_API_URL) |

---

## Verification Checklist

1. `terraform validate` — no errors
2. `terraform apply` — all resources created
3. `curl <LAMBDA_URL>/api/v1/health` — returns `{"status":"healthy"}`
4. `curl <CLOUDFRONT_URL>/` — returns HTML
5. Open CloudFront URL in browser, submit a prediction — works end-to-end
6. Check CloudFront console — distribution status "Deployed"
7. Check CloudWatch — Lambda invocations logging correctly
