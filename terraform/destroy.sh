#!/bin/bash

# Azure Demand Forecasting App - Destruction Script
# This script safely destroys all Azure resources

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Azure CLI
    if ! command_exists az; then
        print_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Terraform
    if ! command_exists terraform; then
        print_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged into Azure
    if ! az account show >/dev/null 2>&1; then
        print_error "Not logged into Azure. Please run 'az login' first."
        exit 1
    fi
    
    print_success "All prerequisites are met!"
}

# Function to show what will be destroyed
show_destruction_plan() {
    print_status "Planning destruction..."
    terraform plan -destroy
    print_success "Destruction plan created"
}

# Function to confirm destruction
confirm_destruction() {
    echo ""
    print_warning "=========================================="
    print_warning "DESTRUCTION CONFIRMATION"
    print_warning "=========================================="
    print_warning "This will PERMANENTLY DELETE all resources:"
    print_warning "- Resource Group and all resources within it"
    print_warning "- Storage Account and ALL DATA"
    print_warning "- Container Registry and images"
    print_warning "- Virtual Network and security groups"
    print_warning "- All monitoring and logging data"
    print_warning ""
    print_warning "THIS ACTION CANNOT BE UNDONE!"
    print_warning "=========================================="
    echo ""
    
    print_warning "Type 'DELETE' to confirm destruction:"
    read -r confirmation
    
    if [ "$confirmation" != "DELETE" ]; then
        print_status "Destruction cancelled by user"
        exit 0
    fi
    
    print_warning "Are you absolutely sure? Type 'YES' to proceed:"
    read -r final_confirmation
    
    if [ "$final_confirmation" != "YES" ]; then
        print_status "Destruction cancelled by user"
        exit 0
    fi
}

# Function to destroy resources
destroy_resources() {
    print_status "Destroying resources..."
    terraform destroy -auto-approve
    print_success "All resources destroyed successfully"
}

# Function to cleanup local files
cleanup_local_files() {
    print_status "Cleaning up local files..."
    
    # Remove Terraform state files
    if [ -f "terraform.tfstate" ]; then
        rm terraform.tfstate
        print_status "Removed terraform.tfstate"
    fi
    
    if [ -f "terraform.tfstate.backup" ]; then
        rm terraform.tfstate.backup
        print_status "Removed terraform.tfstate.backup"
    fi
    
    if [ -f "tfplan" ]; then
        rm tfplan
        print_status "Removed tfplan"
    fi
    
    # Remove .terraform directory
    if [ -d ".terraform" ]; then
        rm -rf .terraform
        print_status "Removed .terraform directory"
    fi
    
    print_success "Local files cleaned up"
}

# Function to show final message
show_final_message() {
    print_success "Destruction completed successfully!"
    echo ""
    print_status "All Azure resources have been permanently deleted."
    print_status "Local Terraform files have been cleaned up."
    echo ""
    print_warning "Remember to:"
    print_warning "- Remove any DNS records if you created custom domains"
    print_warning "- Update any CI/CD pipelines that referenced these resources"
    print_warning "- Remove any local backups if you no longer need them"
    echo ""
}

# Main destruction function
main() {
    echo "=========================================="
    echo "Azure Demand Forecasting App Destruction"
    echo "=========================================="
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Show destruction plan
    show_destruction_plan
    
    # Confirm destruction
    confirm_destruction
    
    # Destroy resources
    destroy_resources
    
    # Cleanup local files
    cleanup_local_files
    
    # Show final message
    show_final_message
}

# Handle script arguments
case "${1:-}" in
    "plan")
        check_prerequisites
        show_destruction_plan
        ;;
    "confirm")
        check_prerequisites
        show_destruction_plan
        confirm_destruction
        destroy_resources
        cleanup_local_files
        show_final_message
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  - Full destruction process with confirmation"
        echo "  plan       - Plan destruction only (no actual destruction)"
        echo "  confirm    - Skip confirmation prompts (DANGEROUS!)"
        echo "  help       - Show this help message"
        echo ""
        echo "WARNING: This will permanently delete all resources and data!"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
