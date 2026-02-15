# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Clarividex seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please do NOT:
- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed

### Please DO:
- Email your findings to [abhismail998@gmail.com]
- Provide sufficient information to reproduce the vulnerability
- Allow reasonable time for us to respond and fix the issue

### What to include in your report:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability

## Security Best Practices for Users

### API Keys
- **Never commit API keys** to version control
- Use environment variables for all sensitive configuration
- Rotate API keys regularly
- Use the minimum required permissions for API keys

### Deployment
- Always use HTTPS in production
- Keep dependencies updated
- Use strong, unique passwords for databases
- Enable rate limiting to prevent abuse
- Store infrastructure IDs (S3 bucket, CloudFront distribution, Lambda function name) in GitHub Secrets, not hardcoded in workflow files

### Data
- This application processes financial data - never store real trading credentials
- Be cautious with any personal financial information
- Review logs to ensure no sensitive data is being logged

### Output Guardrails
Clarividex includes built-in output safety:
- **PII Redaction**: Regex-based detection and redaction of emails, phone numbers, SSNs, and credit card numbers in AI responses
- **Financial Advice Detection**: Flags language that could be construed as personalized financial advice; injects disclaimer
- **Probability Bounds**: All predictions clamped to 15-85% range — we never claim certainty
- **Response Quality**: Ensures AI responses meet minimum quality thresholds

## Acknowledgments

We appreciate the security research community's efforts in helping keep Clarividex and its users safe. Responsible disclosure of vulnerabilities helps us ensure the security and privacy of all users.
