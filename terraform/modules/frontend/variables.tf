# =============================================================================
# Frontend Module - Variables
# =============================================================================

variable "app_name" {
  description = "Application name (used for resource naming)"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "lambda_function_url" {
  description = "Lambda Function URL for API origin"
  type        = string
}
