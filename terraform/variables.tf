# =============================================================================
# Clarividex - Terraform Variables
# =============================================================================

# -----------------------------------------------------------------------------
# AWS Configuration
# -----------------------------------------------------------------------------
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "aws_access_key" {
  description = "AWS access key ID"
  type        = string
  sensitive   = true
}

variable "aws_secret_key" {
  description = "AWS secret access key"
  type        = string
  sensitive   = true
}

# -----------------------------------------------------------------------------
# Application Configuration
# -----------------------------------------------------------------------------
variable "app_name" {
  description = "Application name (used for resource naming)"
  type        = string
  default     = "clarividex"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# -----------------------------------------------------------------------------
# Application Secrets
# -----------------------------------------------------------------------------
variable "gemini_api_key" {
  description = "Google Gemini API key"
  type        = string
  sensitive   = true
}

variable "app_secret_key" {
  description = "Application secret key for sessions/tokens"
  type        = string
  sensitive   = true
}

variable "finnhub_api_key" {
  description = "Finnhub API key (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "gemini_model" {
  description = "Gemini model to use"
  type        = string
  default     = "gemini-2.0-flash"
}

# -----------------------------------------------------------------------------
# Monitoring
# -----------------------------------------------------------------------------
variable "alarm_email" {
  description = "Email address for CloudWatch alarm notifications"
  type        = string
}

