# =============================================================================
# Warmup Module - CloudWatch EventBridge (ping every 5 min)
# =============================================================================

# EventBridge rule: fire every 5 minutes
resource "aws_cloudwatch_event_rule" "warmup" {
  name                = "${var.app_name}-warmup"
  description         = "Ping Lambda every 5 minutes to prevent cold starts"
  schedule_expression = "rate(5 minutes)"
}

# Target: invoke the Lambda function
resource "aws_cloudwatch_event_target" "warmup" {
  rule = aws_cloudwatch_event_rule.warmup.name
  arn  = var.lambda_function_arn

  input = jsonencode({
    httpMethod            = "GET"
    path                  = "/api/v1/health"
    requestContext        = { http = { method = "GET", path = "/api/v1/health" } }
    rawPath               = "/api/v1/health"
    routeKey              = "GET /api/v1/health"
    version               = "2.0"
    isBase64Encoded       = false
    headers               = { "user-agent" = "CloudWatch-Warmup" }
    queryStringParameters = {}
    body                  = null
  })
}

# Permission: allow EventBridge to invoke Lambda
resource "aws_lambda_permission" "warmup" {
  statement_id  = "AllowEventBridgeWarmup"
  action        = "lambda:InvokeFunction"
  function_name = var.lambda_function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.warmup.arn
}
