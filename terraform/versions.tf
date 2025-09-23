# Azure Demand Forecasting App - Provider Versions
# This file specifies the required Terraform and provider versions

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
