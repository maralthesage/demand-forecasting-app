# Azure Demand Forecasting App - Variables
# This file defines all the input variables for the Terraform configuration

variable "project_name" {
  description = "Name of the project (used for resource naming)"
  type        = string
  default     = "demand-forecast"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "location" {
  description = "Azure region where resources will be created"
  type        = string
  default     = "East US"
  
  validation {
    condition = contains([
      "East US", "East US 2", "West US", "West US 2", "West US 3",
      "Central US", "North Central US", "South Central US",
      "West Europe", "North Europe", "UK South", "UK West",
      "Southeast Asia", "East Asia", "Australia East", "Australia Southeast",
      "Canada Central", "Canada East", "Brazil South", "Japan East", "Japan West"
    ], var.location)
    error_message = "Location must be a valid Azure region."
  }
}

variable "container_cpu" {
  description = "CPU cores for the container instance"
  type        = number
  default     = 2
  
  validation {
    condition     = var.container_cpu >= 1 && var.container_cpu <= 4
    error_message = "Container CPU must be between 1 and 4 cores."
  }
}

variable "container_memory" {
  description = "Memory in GB for the container instance"
  type        = number
  default     = 4
  
  validation {
    condition     = var.container_memory >= 1 && var.container_memory <= 16
    error_message = "Container memory must be between 1 and 16 GB."
  }
}

variable "allowed_ssh_ips" {
  description = "IP addresses or CIDR blocks allowed for SSH access"
  type        = string
  default     = "0.0.0.0/0"  # Change this to your specific IP for security
}

variable "create_dns_zone" {
  description = "Whether to create a DNS zone for custom domain"
  type        = bool
  default     = false
}

variable "dns_zone_name" {
  description = "DNS zone name (e.g., example.com)"
  type        = string
  default     = ""
}

variable "dns_record_name" {
  description = "DNS A record name (e.g., app for app.example.com)"
  type        = string
  default     = "app"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "Demand Forecasting"
    Environment = "Production"
    ManagedBy   = "Terraform"
    Owner       = "Data Team"
  }
}

# Advanced Configuration Variables
variable "storage_account_tier" {
  description = "Storage account tier (Standard or Premium)"
  type        = string
  default     = "Standard"
  
  validation {
    condition     = contains(["Standard", "Premium"], var.storage_account_tier)
    error_message = "Storage account tier must be Standard or Premium."
  }
}

variable "storage_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "LRS"
  
  validation {
    condition     = contains(["LRS", "GRS", "RAGRS", "ZRS"], var.storage_replication_type)
    error_message = "Storage replication type must be one of: LRS, GRS, RAGRS, ZRS."
  }
}

variable "container_registry_sku" {
  description = "Container Registry SKU (Basic, Standard, Premium)"
  type        = string
  default     = "Basic"
  
  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.container_registry_sku)
    error_message = "Container Registry SKU must be Basic, Standard, or Premium."
  }
}

variable "log_retention_days" {
  description = "Number of days to retain logs"
  type        = number
  default     = 30
  
  validation {
    condition     = var.log_retention_days >= 7 && var.log_retention_days <= 730
    error_message = "Log retention days must be between 7 and 730."
  }
}

variable "enable_monitoring" {
  description = "Enable Application Insights monitoring"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable backup for storage account"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention days must be between 1 and 365."
  }
}

# Security Variables
variable "enable_private_endpoints" {
  description = "Enable private endpoints for storage and container registry"
  type        = bool
  default     = false
}

variable "enable_network_security_group" {
  description = "Enable Network Security Group rules"
  type        = bool
  default     = true
}

variable "allowed_inbound_ports" {
  description = "List of ports to allow inbound traffic"
  type        = list(number)
  default     = [8501, 443]
}

# Cost Optimization Variables
variable "auto_shutdown_enabled" {
  description = "Enable auto-shutdown for development environments"
  type        = bool
  default     = false
}

variable "auto_shutdown_time" {
  description = "Time for auto-shutdown (HH:MM format)"
  type        = string
  default     = "18:00"
  
  validation {
    condition     = can(regex("^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$", var.auto_shutdown_time))
    error_message = "Auto-shutdown time must be in HH:MM format."
  }
}
