variable "app_name" {
  description = "Application name"
  type        = string
}

variable "lambda_function_arn" {
  description = "Lambda function ARN to invoke"
  type        = string
}

variable "lambda_function_name" {
  description = "Lambda function name"
  type        = string
}
