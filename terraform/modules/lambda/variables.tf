variable "app_name" {
  description = "Application name"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "ecr_repo_url" {
  description = "ECR repository URL"
  type        = string
}

variable "ecr_repo_arn" {
  description = "ECR repository ARN"
  type        = string
}

variable "frontend_url" {
  description = "Frontend URL for CORS"
  type        = string
}

variable "gemini_api_key_ssm_arn" {
  description = "ARN of the Gemini API key SSM parameter"
  type        = string
}

variable "app_secret_key_ssm_arn" {
  description = "ARN of the app secret key SSM parameter"
  type        = string
}

variable "ssm_parameter_arns" {
  description = "List of all SSM parameter ARNs the Lambda can read"
  type        = list(string)
}

variable "gemini_api_key" {
  description = "Gemini API key (passed as env var)"
  type        = string
  sensitive   = true
}

variable "app_secret_key" {
  description = "App secret key (passed as env var)"
  type        = string
  sensitive   = true
}

variable "finnhub_api_key" {
  description = "Finnhub API key (passed as env var)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "gemini_model" {
  description = "Gemini model name"
  type        = string
  default     = "gemini-2.0-flash"
}
