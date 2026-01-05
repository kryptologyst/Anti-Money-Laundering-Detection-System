#!/usr/bin/env python3
"""Setup script for AML Detection System."""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible.
    
    Returns:
        True if Python version is compatible
    """
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.10+")
        return False


def setup_environment() -> bool:
    """Setup the development environment.
    
    Returns:
        True if setup succeeded
    """
    print("🚀 Setting up AML Detection System...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    # Determine activation script based on OS
    if platform.system() == "Windows":
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_command} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Install development dependencies
    if not run_command(f"{pip_command} install pytest pytest-cov black ruff mypy", "Installing development dependencies"):
        return False
    
    print("✅ Environment setup completed successfully!")
    print(f"\nTo activate the virtual environment, run:")
    if platform.system() == "Windows":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    
    return True


def generate_data() -> bool:
    """Generate sample data.
    
    Returns:
        True if data generation succeeded
    """
    print("\n📊 Generating sample data...")
    
    # Determine pip command based on OS
    if platform.system() == "Windows":
        python_command = "venv\\Scripts\\python"
    else:
        python_command = "venv/bin/python"
    
    if not run_command(f"{python_command} scripts/generate_data.py", "Generating sample data"):
        return False
    
    # Verify data was created
    data_files = ["customers.csv", "transactions.csv", "relationships.csv", "features.csv"]
    for file in data_files:
        if not Path(f"data/{file}").exists():
            print(f"❌ Data file {file} not found")
            return False
    
    print("✅ Sample data generated successfully!")
    return True


def run_tests() -> bool:
    """Run tests.
    
    Returns:
        True if tests passed
    """
    print("\n🧪 Running tests...")
    
    # Determine pip command based on OS
    if platform.system() == "Windows":
        python_command = "venv\\Scripts\\python"
    else:
        python_command = "venv/bin/python"
    
    if not run_command(f"{python_command} -m pytest tests/ -v", "Running tests"):
        return False
    
    print("✅ All tests passed!")
    return True


def train_models() -> bool:
    """Train sample models.
    
    Returns:
        True if model training succeeded
    """
    print("\n🤖 Training sample models...")
    
    # Determine pip command based on OS
    if platform.system() == "Windows":
        python_command = "venv\\Scripts\\python"
    else:
        python_command = "venv/bin/python"
    
    if not run_command(f"{python_command} scripts/train_models.py", "Training models"):
        return False
    
    # Verify models were created
    if not Path("assets/model_results.csv").exists():
        print("❌ Model results not found")
        return False
    
    print("✅ Models trained successfully!")
    return True


def main():
    """Main setup function."""
    print("🔍 AML Detection System Setup")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Setup steps
    steps = [
        ("Environment Setup", setup_environment),
        ("Data Generation", generate_data),
        ("Run Tests", run_tests),
        ("Train Models", train_models)
    ]
    
    success_count = 0
    for step_name, step_function in steps:
        print(f"\n📋 {step_name}")
        print("-" * 30)
        
        if step_function():
            success_count += 1
        else:
            print(f"❌ {step_name} failed. Please check the error messages above.")
            break
    
    # Final summary
    print("\n" + "=" * 50)
    if success_count == len(steps):
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if platform.system() == "Windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Run the Streamlit demo:")
        print("   streamlit run demo/aml_dashboard.py")
        print("3. Explore the generated data in the 'data/' directory")
        print("4. Check model results in the 'assets/' directory")
        print("\n⚠️  Remember: This is for research and educational purposes only!")
        print("   NOT for real-world AML compliance or investment advice.")
    else:
        print(f"❌ Setup failed at step {success_count + 1}")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
