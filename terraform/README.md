# Azure Deployment with Terraform

This directory contains Terraform configurations to deploy the Demand Forecasting Application to Azure.

## 🏗️ Architecture

The Terraform configuration creates the following Azure resources:

- **Resource Group**: Container for all resources
- **Storage Account**: For CSV data files and application data
- **Container Registry**: For storing Docker images
- **Container Instance**: Runs the Streamlit application
- **Virtual Network**: Network isolation and security
- **Log Analytics**: Application monitoring and logging
- **Application Insights**: Performance monitoring
- **Public IP**: External access to the application

## 📋 Prerequisites

1. **Azure CLI** installed and configured
2. **Terraform** (version >= 1.0) installed
3. **Docker** installed (for building images)
4. **Azure subscription** with appropriate permissions

## 🚀 Quick Start

### 1. Configure Azure CLI

```bash
# Login to Azure
az login

# Set your subscription
az account set --subscription "Your Subscription ID"

# Verify your account
az account show
```

### 2. Configure Terraform Variables

```bash
# Copy the example variables file
cp terraform.tfvars.example terraform.tfvars

# Edit the variables file with your values
nano terraform.tfvars
```

### 3. Initialize and Deploy

```bash
# Navigate to terraform directory
cd terraform

# Initialize Terraform
terraform init

# Review the deployment plan
terraform plan

# Deploy the infrastructure
terraform apply
```

### 4. Build and Push Docker Image

```bash
# Get the container registry login server
terraform output container_registry_login_server

# Login to Azure Container Registry
az acr login --name <registry-name>

# Build and push the Docker image
docker build -t <registry-name>.azurecr.io/demand-forecasting-app:latest .
docker push <registry-name>.azurecr.io/demand-forecasting-app:latest
```

### 5. Access Your Application

```bash
# Get the application URL
terraform output application_url
```

## 📁 File Structure

```
terraform/
├── main.tf                 # Main Terraform configuration
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── terraform.tfvars.example # Example variables file
└── README.md              # This file
```

## ⚙️ Configuration Options

### Basic Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `project_name` | Project name for resource naming | `demand-forecast` | Any lowercase string |
| `environment` | Environment name | `prod` | `dev`, `staging`, `prod` |
| `location` | Azure region | `East US` | Any valid Azure region |
| `container_cpu` | CPU cores for container | `2` | 1-4 |
| `container_memory` | Memory in GB for container | `4` | 1-16 |

### Security Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `allowed_ssh_ips` | IP addresses allowed for SSH | `0.0.0.0/0` |
| `enable_private_endpoints` | Enable private endpoints | `false` |
| `enable_network_security_group` | Enable NSG rules | `true` |

### Storage Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `storage_account_tier` | Storage account tier | `Standard` |
| `storage_replication_type` | Replication type | `LRS` |
| `container_registry_sku` | Container registry SKU | `Basic` |

## 🔧 Customization

### Custom Domain Setup

To use a custom domain:

1. Set `create_dns_zone = true` in `terraform.tfvars`
2. Set `dns_zone_name` to your domain (e.g., `example.com`)
3. Set `dns_record_name` for the subdomain (e.g., `app`)
4. Run `terraform apply`

### Environment-Specific Configurations

#### Development Environment

```hcl
environment = "dev"
container_cpu = 1
container_memory = 2
auto_shutdown_enabled = true
auto_shutdown_time = "18:00"
```

#### Production Environment

```hcl
environment = "prod"
container_cpu = 4
container_memory = 8
storage_replication_type = "GRS"
container_registry_sku = "Premium"
enable_monitoring = true
```

## 📊 Monitoring and Logging

The deployment includes:

- **Application Insights**: Performance monitoring
- **Log Analytics**: Centralized logging
- **Health Checks**: Container health monitoring
- **Metrics**: Resource utilization tracking

Access monitoring through the Azure Portal or use the provided outputs:

```bash
# Get Application Insights connection string
terraform output application_insights_connection_string

# Get Log Analytics workspace ID
terraform output log_analytics_workspace_id
```

## 🔒 Security Best Practices

1. **Restrict SSH Access**: Update `allowed_ssh_ips` to your specific IP range
2. **Use Private Endpoints**: Set `enable_private_endpoints = true` for production
3. **Enable Monitoring**: Keep `enable_monitoring = true`
4. **Regular Updates**: Keep container images updated
5. **Backup Data**: Enable backup with `enable_backup = true`

## 💰 Cost Optimization

### Development Environment

```hcl
# Use smaller resources
container_cpu = 1
container_memory = 2

# Enable auto-shutdown
auto_shutdown_enabled = true
auto_shutdown_time = "18:00"

# Use basic storage
storage_account_tier = "Standard"
storage_replication_type = "LRS"
```

### Production Environment

```hcl
# Use appropriate resources
container_cpu = 4
container_memory = 8

# Use geo-redundant storage
storage_replication_type = "GRS"

# Enable monitoring and backup
enable_monitoring = true
enable_backup = true
```

## 🗂️ Data Management

### Uploading Data

1. **CSV Data**: Upload to the `csv-data` container
2. **Models**: Upload to the `app-data` container

```bash
# Upload CSV data
az storage blob upload-batch \
  --account-name <storage-account> \
  --destination csv-data \
  --source ./data/csv

# Upload models
az storage blob upload-batch \
  --account-name <storage-account> \
  --destination app-data \
  --source ./models
```

### Data Structure

```
Z:/
├── CSV/
│   ├── F01/
│   │   ├── V2SC1010.csv
│   │   ├── V2AR1001.csv
│   │   └── V2AR1002.csv
│   ├── F02/
│   │   └── V2SC1010.csv
│   ├── F03/
│   │   └── V2SC1010.csv
│   └── F04/
│       └── V2SC1010.csv
```

## 🔄 CI/CD Integration

### GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Terraform Init
        run: terraform init
        working-directory: ./terraform
      
      - name: Terraform Plan
        run: terraform plan
        working-directory: ./terraform
      
      - name: Terraform Apply
        run: terraform apply -auto-approve
        working-directory: ./terraform
      
      - name: Build and Push Docker Image
        run: |
          az acr login --name ${{ secrets.ACR_NAME }}
          docker build -t ${{ secrets.ACR_NAME }}.azurecr.io/demand-forecasting-app:latest .
          docker push ${{ secrets.ACR_NAME }}.azurecr.io/demand-forecasting-app:latest
```

## 🛠️ Troubleshooting

### Common Issues

1. **Container Won't Start**
   - Check logs: `az container logs --name <container-name> --resource-group <rg-name>`
   - Verify image exists in registry
   - Check environment variables

2. **Storage Access Issues**
   - Verify storage account keys
   - Check file share permissions
   - Ensure data files are uploaded

3. **Network Issues**
   - Check NSG rules
   - Verify public IP assignment
   - Test connectivity from Azure Portal

### Useful Commands

```bash
# Check container status
az container show --name <container-name> --resource-group <rg-name>

# View container logs
az container logs --name <container-name> --resource-group <rg-name>

# Restart container
az container restart --name <container-name> --resource-group <rg-name>

# Check storage account
az storage account show --name <storage-account> --resource-group <rg-name>
```

## 🧹 Cleanup

To destroy all resources:

```bash
# Review what will be destroyed
terraform plan -destroy

# Destroy all resources
terraform destroy
```

**Warning**: This will permanently delete all data and resources!

## 📞 Support

For issues with this Terraform configuration:

1. Check the troubleshooting section above
2. Review Azure documentation
3. Check Terraform logs for detailed error messages
4. Verify all prerequisites are met

## 📝 License

This Terraform configuration is part of the Demand Forecasting Application project.
