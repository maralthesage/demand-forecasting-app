# Azure Demand Forecasting App - Main Terraform Configuration
# This file deploys the complete infrastructure for the demand forecasting application

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
  
  # Uncomment and configure for remote state storage
  # backend "azurerm" {
  #   resource_group_name  = "tfstate-rg"
  #   storage_account_name = "tfstatestorageaccount"
  #   container_name       = "tfstate"
  #   key                  = "demand-forecasting.tfstate"
  # }
}

# Configure the Azure Provider
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Data source for current client configuration
data "azurerm_client_config" "current" {}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.location
  
  tags = var.tags
}

# Storage Account for data files
resource "azurerm_storage_account" "data" {
  name                     = "${var.project_name}${var.environment}data${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  # Enable versioning for data protection
  blob_properties {
    versioning_enabled = true
    delete_retention_policy {
      days = 30
    }
  }
  
  tags = var.tags
}

# Storage Container for CSV data
resource "azurerm_storage_container" "csv_data" {
  name                  = "csv-data"
  storage_account_name  = azurerm_storage_account.data.name
  container_access_type = "private"
}

# Storage Container for models and cache
resource "azurerm_storage_container" "app_data" {
  name                  = "app-data"
  storage_account_name  = azurerm_storage_account.data.name
  container_access_type = "private"
}

# Container Registry
resource "azurerm_container_registry" "acr" {
  name                = "${var.project_name}${var.environment}acr${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  
  tags = var.tags
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-${var.environment}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  tags = var.tags
}

# Subnet for Container Instances
resource "azurerm_subnet" "container_instances" {
  name                 = "${var.project_name}-${var.environment}-container-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
  
  delegation {
    name = "delegation"
    service_delegation {
      name    = "Microsoft.ContainerInstance.containerGroups"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

# Network Security Group
resource "azurerm_network_security_group" "main" {
  name                = "${var.project_name}-${var.environment}-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  # Allow HTTP traffic
  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1000
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8501"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  # Allow HTTPS traffic
  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  # Allow SSH for debugging (optional)
  security_rule {
    name                       = "AllowSSH"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.allowed_ssh_ips
    destination_address_prefix = "*"
  }
  
  tags = var.tags
}

# Associate NSG with subnet
resource "azurerm_subnet_network_security_group_association" "main" {
  subnet_id                 = azurerm_subnet.container_instances.id
  network_security_group_id = azurerm_network_security_group.main.id
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.project_name}-${var.environment}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = var.tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "${var.project_name}-${var.environment}-insights"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  
  tags = var.tags
}

# Container Instance for the application
resource "azurerm_container_group" "app" {
  name                = "${var.project_name}-${var.environment}-app"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  ip_address_type     = "Public"
  dns_name_label      = "${var.project_name}-${var.environment}-app"
  os_type             = "Linux"
  
  # Use Azure Container Registry
  image_registry_credential {
    server   = azurerm_container_registry.acr.login_server
    username = azurerm_container_registry.acr.admin_username
    password = azurerm_container_registry.acr.admin_password
  }
  
  container {
    name   = "demand-forecasting-app"
    image  = "${azurerm_container_registry.acr.login_server}/demand-forecasting-app:latest"
    cpu    = var.container_cpu
    memory = var.container_memory
    
    ports {
      port     = 8501
      protocol = "TCP"
    }
    
    # Environment variables
    environment_variables = {
      SALES_FORECAST_DATA_PATH = "/mnt/data"
      STREAMLIT_SERVER_PORT    = "8501"
      STREAMLIT_SERVER_ADDRESS = "0.0.0.0"
      PYTHONPATH               = "/app"
      LOG_LEVEL                = "INFO"
    }
    
    # Mount Azure File Share for data
    volume {
      name                 = "data-volume"
      mount_path           = "/mnt/data"
      storage_account_name = azurerm_storage_account.data.name
      storage_account_key  = azurerm_storage_account.data.primary_access_key
      share_name           = azurerm_storage_share.data.name
    }
    
    # Mount Azure File Share for models and cache
    volume {
      name                 = "app-volume"
      mount_path           = "/app/cache"
      storage_account_name = azurerm_storage_account.data.name
      storage_account_key  = azurerm_storage_account.data.primary_access_key
      share_name           = azurerm_storage_share.app_data.name
    }
    
    # Health check
    liveness_probe {
      http_get {
        path = "/_stcore/health"
        port = 8501
      }
      initial_delay_seconds = 30
      period_seconds        = 10
      timeout_seconds       = 5
      failure_threshold     = 3
    }
    
    readiness_probe {
      http_get {
        path = "/_stcore/health"
        port = 8501
      }
      initial_delay_seconds = 10
      period_seconds        = 5
      timeout_seconds       = 3
      failure_threshold     = 3
    }
  }
  
  # Restart policy
  restart_policy = "Always"
  
  tags = var.tags
  
  depends_on = [
    azurerm_storage_share.data,
    azurerm_storage_share.app_data
  ]
}

# Azure File Share for CSV data
resource "azurerm_storage_share" "data" {
  name                 = "csv-data"
  storage_account_name = azurerm_storage_account.data.name
  quota                = 100  # GB
}

# Azure File Share for app data (models, cache)
resource "azurerm_storage_share" "app_data" {
  name                 = "app-data"
  storage_account_name = azurerm_storage_account.data.name
  quota                = 50   # GB
}

# Public IP for the container instance
resource "azurerm_public_ip" "app" {
  name                = "${var.project_name}-${var.environment}-app-ip"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"
  
  tags = var.tags
}

# DNS Zone (optional - for custom domain)
resource "azurerm_dns_zone" "main" {
  count               = var.create_dns_zone ? 1 : 0
  name                = var.dns_zone_name
  resource_group_name = azurerm_resource_group.main.name
  
  tags = var.tags
}

# DNS A record (optional)
resource "azurerm_dns_a_record" "app" {
  count               = var.create_dns_zone ? 1 : 0
  name                = var.dns_record_name
  zone_name           = azurerm_dns_zone.main[0].name
  resource_group_name = azurerm_resource_group.main.name
  ttl                 = 300
  records             = [azurerm_public_ip.app.ip_address]
}
