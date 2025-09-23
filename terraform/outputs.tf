# Azure Demand Forecasting App - Outputs
# This file defines all the output values from the Terraform configuration

output "resource_group_name" {
  description = "Name of the created resource group"
  value       = azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "Location of the created resource group"
  value       = azurerm_resource_group.main.location
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.data.name
}

output "storage_account_primary_endpoint" {
  description = "Primary endpoint of the storage account"
  value       = azurerm_storage_account.data.primary_blob_endpoint
}

output "container_registry_name" {
  description = "Name of the container registry"
  value       = azurerm_container_registry.acr.name
}

output "container_registry_login_server" {
  description = "Login server URL of the container registry"
  value       = azurerm_container_registry.acr.login_server
}

output "container_registry_admin_username" {
  description = "Admin username for the container registry"
  value       = azurerm_container_registry.acr.admin_username
  sensitive   = true
}

output "container_registry_admin_password" {
  description = "Admin password for the container registry"
  value       = azurerm_container_registry.acr.admin_password
  sensitive   = true
}

output "container_instance_name" {
  description = "Name of the container instance"
  value       = azurerm_container_group.app.name
}

output "container_instance_fqdn" {
  description = "FQDN of the container instance"
  value       = azurerm_container_group.app.fqdn
}

output "application_url" {
  description = "URL to access the demand forecasting application"
  value       = "http://${azurerm_container_group.app.fqdn}:8501"
}

output "public_ip_address" {
  description = "Public IP address of the container instance"
  value       = azurerm_public_ip.app.ip_address
}

output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics workspace"
  value       = azurerm_log_analytics_workspace.main.id
}

output "application_insights_instrumentation_key" {
  description = "Instrumentation key for Application Insights"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}

output "application_insights_connection_string" {
  description = "Connection string for Application Insights"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "virtual_network_name" {
  description = "Name of the virtual network"
  value       = azurerm_virtual_network.main.name
}

output "subnet_name" {
  description = "Name of the container instances subnet"
  value       = azurerm_subnet.container_instances.name
}

output "network_security_group_name" {
  description = "Name of the network security group"
  value       = azurerm_network_security_group.main.name
}

# Data Storage Information
output "csv_data_container_name" {
  description = "Name of the CSV data storage container"
  value       = azurerm_storage_container.csv_data.name
}

output "app_data_container_name" {
  description = "Name of the app data storage container"
  value       = azurerm_storage_container.app_data.name
}

output "csv_data_share_name" {
  description = "Name of the CSV data file share"
  value       = azurerm_storage_share.data.name
}

output "app_data_share_name" {
  description = "Name of the app data file share"
  value       = azurerm_storage_share.app_data.name
}

# DNS Information (if created)
output "dns_zone_name" {
  description = "Name of the DNS zone (if created)"
  value       = var.create_dns_zone ? azurerm_dns_zone.main[0].name : null
}

output "dns_record_name" {
  description = "Name of the DNS A record (if created)"
  value       = var.create_dns_zone ? azurerm_dns_a_record.app[0].name : null
}

output "custom_domain_url" {
  description = "Custom domain URL (if DNS zone is created)"
  value       = var.create_dns_zone ? "https://${var.dns_record_name}.${var.dns_zone_name}" : null
}

# Deployment Information
output "deployment_summary" {
  description = "Summary of the deployment"
  value = {
    project_name        = var.project_name
    environment         = var.environment
    location            = var.location
    application_url     = "http://${azurerm_container_group.app.fqdn}:8501"
    container_cpu       = var.container_cpu
    container_memory    = var.container_memory
    storage_account     = azurerm_storage_account.data.name
    container_registry  = azurerm_container_registry.acr.name
    monitoring_enabled  = var.enable_monitoring
    backup_enabled      = var.enable_backup
  }
}

# Connection Information for Data Upload
output "data_upload_instructions" {
  description = "Instructions for uploading data to the storage account"
  value = {
    storage_account_name = azurerm_storage_account.data.name
    csv_data_container   = azurerm_storage_container.csv_data.name
    app_data_container   = azurerm_storage_container.app_data.name
    upload_command = "az storage blob upload-batch --account-name ${azurerm_storage_account.data.name} --destination ${azurerm_storage_container.csv_data.name} --source ./data"
  }
}

# Security Information
output "security_notes" {
  description = "Important security information"
  value = {
    ssh_allowed_ips     = var.allowed_ssh_ips
    public_ip_address   = azurerm_public_ip.app.ip_address
    application_port    = 8501
    security_group      = azurerm_network_security_group.main.name
    private_endpoints   = var.enable_private_endpoints
  }
}
