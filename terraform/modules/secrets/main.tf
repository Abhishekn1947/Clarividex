# =============================================================================
# Secrets Module - SSM Parameter Store
# =============================================================================

resource "aws_ssm_parameter" "gemini_api_key" {
  name        = "/${var.app_name}/${var.environment}/gemini-api-key"
  description = "Google Gemini API key"
  type        = "SecureString"
  value       = var.gemini_api_key

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "app_secret_key" {
  name        = "/${var.app_name}/${var.environment}/app-secret-key"
  description = "Application secret key"
  type        = "SecureString"
  value       = var.app_secret_key

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "finnhub_api_key" {
  name        = "/${var.app_name}/${var.environment}/finnhub-api-key"
  description = "Finnhub API key"
  type        = "SecureString"
  value       = var.finnhub_api_key != "" ? var.finnhub_api_key : "not-configured"

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "app_environment" {
  name        = "/${var.app_name}/${var.environment}/app-env"
  description = "Application environment"
  type        = "String"
  value       = var.environment
}

resource "aws_ssm_parameter" "app_name" {
  name        = "/${var.app_name}/${var.environment}/app-name"
  description = "Application name"
  type        = "String"
  value       = var.app_name
}
