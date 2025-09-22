#!/bin/bash

# Azure Demand Forecasting App - Deployment Script
# This script automates the deployment process

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
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged into Azure
    if ! az account show >/dev/null 2>&1; then
        print_error "Not logged into Azure. Please run 'az login' first."
        exit 1
    fi
    
    print_success "All prerequisites are met!"
}

# Function to setup variables file
setup_variables() {
    print_status "Setting up variables file..."
    
    if [ ! -f "terraform.tfvars" ]; then
        if [ -f "terraform.tfvars.example" ]; then
            cp terraform.tfvars.example terraform.tfvars
            print_warning "Created terraform.tfvars from example. Please review and customize the values."
            print_warning "Press Enter to continue after reviewing terraform.tfvars..."
            read -r
        else
            print_error "terraform.tfvars.example not found!"
            exit 1
        fi
    else
        print_success "terraform.tfvars already exists"
    fi
}

# Function to initialize Terraform
init_terraform() {
    print_status "Initializing Terraform..."
    terraform init
    print_success "Terraform initialized successfully"
}

# Function to plan deployment
plan_deployment() {
    print_status "Planning deployment..."
    terraform plan -out=tfplan
    print_success "Deployment plan created"
}

# Function to apply deployment
apply_deployment() {
    print_status "Applying deployment..."
    terraform apply tfplan
    print_success "Infrastructure deployed successfully"
}

# Function to get outputs
get_outputs() {
    print_status "Getting deployment outputs..."
    
    # Get key outputs
    REGISTRY_NAME=$(terraform output -raw container_registry_name)
    APP_URL=$(terraform output -raw application_url)
    STORAGE_ACCOUNT=$(terraform output -raw storage_account_name)
    
    print_success "Deployment outputs retrieved"
    echo ""
    print_status "Key Information:"
    echo "  Container Registry: $REGISTRY_NAME"
    echo "  Application URL: $APP_URL"
    echo "  Storage Account: $STORAGE_ACCOUNT"
    echo ""
}

# Function to build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Get registry name
    REGISTRY_NAME=$(terraform output -raw container_registry_name)
    REGISTRY_URL=$(terraform output -raw container_registry_login_server)
    
    # Login to registry
    print_status "Logging into Azure Container Registry..."
    az acr login --name "$REGISTRY_NAME"
    
    # Build image
    print_status "Building Docker image..."
    docker build -t "$REGISTRY_URL/demand-forecasting-app:latest" ..
    
    # Push image
    print_status "Pushing Docker image..."
    docker push "$REGISTRY_URL/demand-forecasting-app:latest"
    
    print_success "Docker image built and pushed successfully"
}

# Function to restart container
restart_container() {
    print_status "Restarting container instance..."
    
    # Get resource names
    RG_NAME=$(terraform output -raw resource_group_name)
    CONTAINER_NAME=$(terraform output -raw container_instance_name)
    
    # Restart container
    az container restart --name "$CONTAINER_NAME" --resource-group "$RG_NAME"
    
    print_success "Container instance restarted"
}

# Function to show final information
show_final_info() {
    print_success "Deployment completed successfully!"
    echo ""
    print_status "Next Steps:"
    echo "1. Upload your data files to the storage account"
    echo "2. Access your application at: $(terraform output -raw application_url)"
    echo "3. Monitor your application in the Azure Portal"
    echo ""
    print_status "Useful Commands:"
    echo "  View logs: az container logs --name $(terraform output -raw container_instance_name) --resource-group $(terraform output -raw resource_group_name)"
    echo "  Restart app: az container restart --name $(terraform output -raw container_instance_name) --resource-group $(terraform output -raw resource_group_name)"
    echo "  View outputs: terraform output"
    echo ""
}

# Function to upload sample data
upload_sample_data() {
    print_status "Would you like to upload sample data? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Uploading sample data..."
        
        # Get storage account name
        STORAGE_ACCOUNT=$(terraform output -raw storage_account_name)
        
        # Create sample data directory structure
        mkdir -p sample_data/CSV/F01
        mkdir -p sample_data/CSV/F02
        mkdir -p sample_data/CSV/F03
        mkdir -p sample_data/CSV/F04
        
        # Create sample CSV files (you can replace these with your actual data)
        echo "NUMMER,BANAME1" > sample_data/CSV/F01/V2AR1002.csv
        echo "PROD001,Sample Product 1" >> sample_data/CSV/F01/V2AR1002.csv
        echo "PROD002,Sample Product 2" >> sample_data/CSV/F01/V2AR1002.csv
        
        # Upload to storage
        az storage blob upload-batch \
            --account-name "$STORAGE_ACCOUNT" \
            --destination csv-data \
            --source sample_data \
            --overwrite
        
        # Clean up
        rm -rf sample_data
        
        print_success "Sample data uploaded successfully"
    fi
}

# Main deployment function
main() {
    echo "=========================================="
    echo "Azure Demand Forecasting App Deployment"
    echo "=========================================="
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Setup variables
    setup_variables
    
    # Initialize Terraform
    init_terraform
    
    # Plan deployment
    plan_deployment
    
    # Ask for confirmation
    print_warning "Do you want to proceed with the deployment? (y/n)"
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_status "Deployment cancelled by user"
        exit 0
    fi
    
    # Apply deployment
    apply_deployment
    
    # Get outputs
    get_outputs
    
    # Build and push Docker image
    build_and_push_image
    
    # Restart container
    restart_container
    
    # Upload sample data
    upload_sample_data
    
    # Show final information
    show_final_info
}

# Handle script arguments
case "${1:-}" in
    "plan")
        check_prerequisites
        setup_variables
        init_terraform
        plan_deployment
        ;;
    "apply")
        check_prerequisites
        setup_variables
        init_terraform
        apply_deployment
        get_outputs
        ;;
    "build")
        check_prerequisites
        build_and_push_image
        restart_container
        ;;
    "restart")
        restart_container
        ;;
    "outputs")
        get_outputs
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  - Full deployment process"
        echo "  plan       - Plan deployment only"
        echo "  apply      - Apply deployment only"
        echo "  build      - Build and push Docker image only"
        echo "  restart    - Restart container instance only"
        echo "  outputs    - Show deployment outputs only"
        echo "  help       - Show this help message"
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
