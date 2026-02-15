# =============================================================================
# Monitoring Module - CloudWatch Alarms + SNS
# =============================================================================

# -----------------------------------------------------------------------------
# SNS Topic for alarm notifications
# -----------------------------------------------------------------------------
resource "aws_sns_topic" "alarms" {
  name = "${var.app_name}-${var.environment}-alarms"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# -----------------------------------------------------------------------------
# Alarm 1: Lambda Errors (Sum > 5 in 5 min)
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.app_name}-lambda-errors"
  alarm_description   = "Lambda function errors exceed threshold"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = var.lambda_function_name
  }

  alarm_actions = [aws_sns_topic.alarms.arn]
  ok_actions    = [aws_sns_topic.alarms.arn]
}

# -----------------------------------------------------------------------------
# Alarm 2: Lambda Duration (Avg > 60s over 3 periods)
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  alarm_name          = "${var.app_name}-lambda-duration"
  alarm_description   = "Lambda function duration exceeds 60 seconds average"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Average"
  threshold           = 60000  # 60 seconds in milliseconds
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = var.lambda_function_name
  }

  alarm_actions = [aws_sns_topic.alarms.arn]
}

# -----------------------------------------------------------------------------
# Alarm 3: Lambda Throttles (Sum > 3 in 5 min)
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  alarm_name          = "${var.app_name}-lambda-throttles"
  alarm_description   = "Lambda function is being throttled"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 3
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = var.lambda_function_name
  }

  alarm_actions = [aws_sns_topic.alarms.arn]
}

# -----------------------------------------------------------------------------
# Alarm 4: Lambda 5xx Errors (via log metric filter)
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_metric_filter" "lambda_5xx" {
  name           = "${var.app_name}-5xx-errors"
  log_group_name = "/aws/lambda/${var.lambda_function_name}"
  pattern        = "\"500 Internal Server Error\""

  metric_transformation {
    name          = "${var.app_name}-5xx-count"
    namespace     = "Custom/${var.app_name}"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_metric_alarm" "lambda_5xx" {
  alarm_name          = "${var.app_name}-5xx-errors"
  alarm_description   = "5xx errors detected in Lambda logs"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "${var.app_name}-5xx-count"
  namespace           = "Custom/${var.app_name}"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alarms.arn]
}

# -----------------------------------------------------------------------------
# Alarm 5: Concurrent Executions (Max > 50)
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "lambda_concurrency" {
  alarm_name          = "${var.app_name}-high-concurrency"
  alarm_description   = "Lambda concurrent executions exceed threshold"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ConcurrentExecutions"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Maximum"
  threshold           = 50
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = var.lambda_function_name
  }

  alarm_actions = [aws_sns_topic.alarms.arn]
}
