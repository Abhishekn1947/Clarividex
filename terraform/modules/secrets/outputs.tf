output "gemini_api_key_arn" {
  description = "ARN of the Gemini API key parameter"
  value       = aws_ssm_parameter.gemini_api_key.arn
}

output "app_secret_key_arn" {
  description = "ARN of the app secret key parameter"
  value       = aws_ssm_parameter.app_secret_key.arn
}

output "all_parameter_arns" {
  description = "List of all SSM parameter ARNs"
  value = [
    aws_ssm_parameter.gemini_api_key.arn,
    aws_ssm_parameter.app_secret_key.arn,
    aws_ssm_parameter.finnhub_api_key.arn,
    aws_ssm_parameter.app_environment.arn,
    aws_ssm_parameter.app_name.arn,
  ]
}
