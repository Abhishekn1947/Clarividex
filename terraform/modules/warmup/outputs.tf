output "event_rule_arn" {
  description = "EventBridge warmup rule ARN"
  value       = aws_cloudwatch_event_rule.warmup.arn
}

output "event_rule_name" {
  description = "EventBridge warmup rule name"
  value       = aws_cloudwatch_event_rule.warmup.name
}
