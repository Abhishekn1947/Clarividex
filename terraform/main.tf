# =============================================================================
# Clarividex - Terraform Root Configuration
# AWS Serverless Deployment (Lambda + ECR, no VPC)
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region     = var.aws_region
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key

  default_tags {
    tags = {
      Project     = var.app_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# -----------------------------------------------------------------------------
# ECR - Container Registry
# -----------------------------------------------------------------------------
module "ecr" {
  source   = "./modules/ecr"
  app_name = var.app_name
}

# -----------------------------------------------------------------------------
# Secrets - SSM Parameter Store
# -----------------------------------------------------------------------------
module "secrets" {
  source         = "./modules/secrets"
  app_name       = var.app_name
  environment    = var.environment
  gemini_api_key = var.gemini_api_key
  app_secret_key = var.app_secret_key
  finnhub_api_key = var.finnhub_api_key
}

# -----------------------------------------------------------------------------
# Lambda - Function + Function URL + IAM
# -----------------------------------------------------------------------------
module "lambda" {
  source = "./modules/lambda"

  app_name       = var.app_name
  environment    = var.environment
  ecr_repo_url   = module.ecr.repository_url
  ecr_repo_arn   = module.ecr.repository_arn
  frontend_url   = module.frontend.cloudfront_url
  gemini_api_key_ssm_arn = module.secrets.gemini_api_key_arn
  app_secret_key_ssm_arn = module.secrets.app_secret_key_arn

  ssm_parameter_arns = module.secrets.all_parameter_arns

  gemini_api_key  = var.gemini_api_key
  app_secret_key  = var.app_secret_key
  finnhub_api_key = var.finnhub_api_key
  gemini_model    = var.gemini_model
}

# -----------------------------------------------------------------------------
# Monitoring - CloudWatch Alarms + SNS
# -----------------------------------------------------------------------------
module "monitoring" {
  source = "./modules/monitoring"

  app_name            = var.app_name
  environment         = var.environment
  alarm_email         = var.alarm_email
  lambda_function_name = module.lambda.function_name
}

# -----------------------------------------------------------------------------
# Frontend - S3 + CloudFront Static Hosting
# -----------------------------------------------------------------------------
module "frontend" {
  source      = "./modules/frontend"
  app_name    = var.app_name
  environment = var.environment
}

# -----------------------------------------------------------------------------
# Warmup - CloudWatch EventBridge (ping every 5 min)
# -----------------------------------------------------------------------------
module "warmup" {
  source = "./modules/warmup"

  app_name            = var.app_name
  lambda_function_arn = module.lambda.function_arn
  lambda_function_name = module.lambda.function_name
}
