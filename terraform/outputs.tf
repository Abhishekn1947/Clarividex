# =============================================================================
# Clarividex - Terraform Outputs
# =============================================================================

output "function_url" {
  description = "Lambda Function URL (public HTTPS endpoint)"
  value       = module.lambda.function_url
}

output "ecr_repository_url" {
  description = "ECR repository URL for Docker push"
  value       = module.ecr.repository_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = module.lambda.function_name
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = module.lambda.function_arn
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alarm notifications"
  value       = module.monitoring.sns_topic_arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = module.lambda.log_group_name
}

output "api_health_url" {
  description = "Health check endpoint"
  value       = "${module.lambda.function_url}api/v1/health"
}

output "frontend_url" {
  description = "Frontend CloudFront URL"
  value       = module.frontend.cloudfront_url
}

output "frontend_s3_bucket" {
  description = "Frontend S3 bucket name"
  value       = module.frontend.s3_bucket_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID (for cache invalidation)"
  value       = module.frontend.cloudfront_distribution_id
}
