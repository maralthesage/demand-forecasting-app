# GitHub Actions Setup Guide

This guide explains how to set up GitHub Actions for the Demand Forecasting App.

## Overview

The CI/CD pipeline includes:
- **Test**: Code linting and basic functionality tests
- **Build**: Docker image creation
- **Data Processing**: Model training and data processing (optional)
- **Deploy**: Staging and production deployments
- **Notify**: Success/failure notifications

## Setup Steps

### 1. Repository Settings

1. Go to your repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Add the following secrets (if needed):

```
SALES_FORECAST_DATA_PATH=/path/to/your/data
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### 2. Environment Protection Rules

1. Go to "Settings" → "Environments"
2. Create `staging` and `production` environments
3. Add protection rules as needed:
   - Required reviewers
   - Wait timer
   - Deployment branches

### 3. Branch Protection

1. Go to "Settings" → "Branches"
2. Add branch protection rules for `main` and `develop` branches
3. Enable "Require status checks to pass before merging"

## Workflow Triggers

- **Push to main/develop**: Runs full CI/CD pipeline
- **Pull Request**: Runs tests only
- **Manual Dispatch**: Allows manual triggering with options
- **Scheduled**: Daily data processing (optional)

## Development Mode

The current workflow is configured for development mode:
- Simulated deployments (no actual infrastructure)
- Basic testing without external dependencies
- Simplified error handling

## Customization

### For Production Deployment

1. Update the deployment steps in `deploy-production` job
2. Add your actual deployment commands (Docker registry, cloud services, etc.)
3. Configure proper secrets and environment variables
4. Set up monitoring and alerting

### For Data Processing

1. Configure data access in repository secrets
2. Update the data processing commands
3. Set up proper error handling and notifications

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **Docker Build Failures**: Check Dockerfile and build context
3. **Test Failures**: Verify test data and environment setup
4. **Permission Errors**: Check repository secrets and permissions

### Debug Mode

To debug issues:
1. Check the Actions tab in your repository
2. Click on the failed workflow run
3. Review the logs for each step
4. Use the "Re-run jobs" feature to retry

## Security Notes

- Never commit secrets to the repository
- Use GitHub Secrets for sensitive data
- Regularly rotate access tokens
- Review and audit workflow permissions
