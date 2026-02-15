# =============================================================================
# Lambda Module - Function + Function URL + IAM
# =============================================================================

# -----------------------------------------------------------------------------
# IAM Role for Lambda
# -----------------------------------------------------------------------------
resource "aws_iam_role" "lambda" {
  name = "${var.app_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# CloudWatch Logs policy
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ECR pull policy
resource "aws_iam_role_policy" "ecr_pull" {
  name = "${var.app_name}-ecr-pull"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = var.ecr_repo_arn
      },
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      }
    ]
  })
}

# SSM Parameter Store read policy
resource "aws_iam_role_policy" "ssm_read" {
  name = "${var.app_name}-ssm-read"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = [
          "ssm:GetParameter",
          "ssm:GetParameters"
        ]
        Resource = var.ssm_parameter_arns
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# CloudWatch Log Group
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.app_name}-${var.environment}"
  retention_in_days = 14
}

# -----------------------------------------------------------------------------
# Lambda Function
# -----------------------------------------------------------------------------
resource "aws_lambda_function" "app" {
  function_name = "${var.app_name}-${var.environment}"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${var.ecr_repo_url}:latest"
  timeout       = 120
  memory_size   = 1536
  architectures = ["x86_64"]

  environment {
    variables = {
      APP_ENV         = "production"
      APP_DEBUG       = "false"
      GEMINI_API_KEY  = var.gemini_api_key
      GEMINI_MODEL    = var.gemini_model
      APP_SECRET_KEY  = var.app_secret_key
      FINNHUB_API_KEY = var.finnhub_api_key
      FRONTEND_URL    = var.frontend_url
      PYTHONPATH      = "/var/task"
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy_attachment.lambda_logs,
  ]

  # Ignore image_uri changes — CI/CD updates this outside Terraform
  lifecycle {
    ignore_changes = [image_uri]
  }
}

# -----------------------------------------------------------------------------
# Lambda Function URL (public HTTPS, no API Gateway needed)
# -----------------------------------------------------------------------------
# Allow public (unauthenticated) access to Function URL
# Newer AWS accounts require BOTH InvokeFunctionUrl and InvokeFunction permissions
resource "aws_lambda_permission" "function_url_public" {
  statement_id           = "AllowPublicFunctionURL"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.app.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "function_url_invoke" {
  statement_id  = "AllowPublicInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.app.function_name
  principal     = "*"
}

resource "aws_lambda_function_url" "app" {
  function_name      = aws_lambda_function.app.function_name
  authorization_type = "NONE"

  # No cors block — CORS is handled entirely by FastAPI CORSMiddleware
  # to avoid duplicate Access-Control-Allow-Origin headers
}
