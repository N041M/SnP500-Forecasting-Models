#!/usr/bin/env python
"""
Setup script to install and verify dependencies for S&P 500 LSTM model
Run this script first to ensure all dependencies are properly installed.
"""

import subprocess
import sys
import platform

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_dependencies():
    """Check and install required dependencies"""
    
    print("=" * 60)
    print("S&P 500 LSTM Model - Dependency Setup")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print("=" * 60)
    
    # Core dependencies
    core_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ]
    
    print("\n1. Installing core packages...")
    for package in core_packages:
        try:
            install_package(package)
            print(f"✓ {package} installed successfully")
        except Exception as e:
            print(f"✗ Error installing {package}: {e}")
    
    # Choose between TensorFlow and PyTorch
    print("\n2. Choose your deep learning framework:")
    print("   [1] TensorFlow (recommended for beginners)")
    print("   [2] PyTorch (more flexible, easier debugging)")
    print("   [3] Both (install both frameworks)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    # Install TensorFlow
    if choice in ["1", "3"]:
        print("\n3. Installing TensorFlow...")
        
        # Check if on Apple Silicon Mac
        if platform.system() == "Darwin" and platform.processor() == "arm":
            print("Detected Apple Silicon Mac. Installing TensorFlow for M1/M2...")
            try:
                install_package("tensorflow-macos")
                install_package("tensorflow-metal")
                print("✓ TensorFlow for Apple Silicon installed successfully")
            except Exception as e:
                print(f"✗ Error installing TensorFlow for Apple Silicon: {e}")
                print("Trying standard TensorFlow installation...")
                try:
                    install_package("tensorflow")
                    print("✓ Standard TensorFlow installed successfully")
                except Exception as e2:
                    print(f"✗ Error installing TensorFlow: {e2}")
        else:
            try:
                install_package("tensorflow")
                print("✓ TensorFlow installed successfully")
            except Exception as e:
                print(f"✗ Error installing TensorFlow: {e}")
    
    # Install PyTorch
    if choice in ["2", "3"]:
        print("\n3. Installing PyTorch...")
        
        # Check for CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                print("CUDA is available. PyTorch with CUDA support is recommended.")
                cuda_choice = input("Install PyTorch with CUDA support? (y/n): ").strip().lower()
                if cuda_choice == 'y':
                    # Install PyTorch with CUDA
                    install_package("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                else:
                    install_package("torch")
            else:
                install_package("torch")
            print("✓ PyTorch installed successfully")
        except ImportError:
            # PyTorch not installed yet
            install_package("torch")
            print("✓ PyTorch installed successfully")
        except Exception as e:
            print(f"✗ Error installing PyTorch: {e}")
    
    # Optional packages
    print("\n4. Install optional packages for enhanced functionality? (y/n)")
    optional_choice = input("   (includes yfinance for real S&P data, plotly for interactive plots): ").strip().lower()
    
    if optional_choice == 'y':
        optional_packages = [
            "yfinance",  # For downloading real S&P 500 data
            "ta",  # Technical analysis indicators
            "plotly"  # Interactive visualizations
        ]
        
        for package in optional_packages:
            try:
                install_package(package)
                print(f"✓ {package} installed successfully")
            except Exception as e:
                print(f"✗ Error installing {package}: {e}")

def verify_installation():
    """Verify that all required packages are installed correctly"""
    
    print("\n" + "=" * 60)
    print("Verifying Installation")
    print("=" * 60)
    
    # Check core packages
    core_imports = {
        "numpy": "np",
        "pandas": "pd",
        "matplotlib.pyplot": "plt",
        "sklearn": "sklearn"
    }
    
    print("\nCore packages:")
    for module, alias in core_imports.items():
        try:
            exec(f"import {module} as {alias}")
            version = eval(f"{alias.split('.')[0]}.__version__")
            print(f"✓ {module}: version {version}")
        except ImportError:
            print(f"✗ {module}: NOT INSTALLED")
        except AttributeError:
            print(f"✓ {module}: installed (version not available)")
    
    # Check TensorFlow
    print("\nDeep Learning Frameworks:")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: version {tf.__version__}")
        if tf.config.list_physical_devices('GPU'):
            print("  - GPU support: AVAILABLE")
        else:
            print("  - GPU support: NOT AVAILABLE")
    except ImportError:
        print("✗ TensorFlow: NOT INSTALLED")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: version {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  - CUDA support: AVAILABLE (CUDA {torch.version.cuda})")
            print(f"  - GPU devices: {torch.cuda.device_count()}")
        else:
            print("  - CUDA support: NOT AVAILABLE")
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
    
    # Check optional packages
    print("\nOptional packages:")
    optional_imports = ["yfinance", "ta", "plotly"]
    
    for module in optional_imports:
        try:
            exec(f"import {module}")
            version = eval(f"{module}.__version__")
            print(f"✓ {module}: version {version}")
        except ImportError:
            print(f"✗ {module}: NOT INSTALLED (optional)")
        except AttributeError:
            print(f"✓ {module}: installed (version not available)")

def create_test_script():
    """Create a simple test script to verify the model can run"""
    
    test_code = '''
import numpy as np
import pandas as pd

# Test data generation
def test_model_import():
    """Test if the model can be imported and basic operations work"""
    
    print("Testing model import and basic operations...")
    
    # Test numpy
    arr = np.random.randn(100, 10)
    print(f"✓ NumPy array created: shape {arr.shape}")
    
    # Test pandas
    df = pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(10)])
    print(f"✓ Pandas DataFrame created: shape {df.shape}")
    
    # Test sklearn
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    print(f"✓ Sklearn MinMaxScaler works: scaled shape {scaled.shape}")
    
    # Try to test deep learning framework
    framework_tested = False
    
    # Try TensorFlow
    try:
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(10, 10)),
            tf.keras.layers.Dense(1)
        ])
        print("✓ TensorFlow LSTM model created successfully")
        framework_tested = True
    except ImportError:
        print("- TensorFlow not available")
    except Exception as e:
        print(f"✗ TensorFlow error: {e}")
    
    # Try PyTorch
    try:
        import torch
        import torch.nn as nn
        
        class SimpleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        
        model = SimpleLSTM()
        print("✓ PyTorch LSTM model created successfully")
        framework_tested = True
    except ImportError:
        print("- PyTorch not available")
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
    
    if framework_tested:
        print("\\n✓ All tests passed! Your environment is ready for the S&P 500 LSTM model.")
    else:
        print("\\n⚠ No deep learning framework available. Please install TensorFlow or PyTorch.")

if __name__ == "__main__":
    test_model_import()
'''
    
    with open("test_environment.py", "w") as f:
        f.write(test_code)
    
    print("\nTest script created: test_environment.py")
    print("Run 'python test_environment.py' to verify your setup.")

if __name__ == "__main__":
    print("\nStarting dependency setup...\n")
    
    # Install dependencies
    check_and_install_dependencies()
    
    # Verify installation
    verify_installation()
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python test_environment.py' to verify everything works")
    print("2. Choose your model implementation:")
    print("   - v1.3.py for TensorFlow version")
    print("   - sp500_lstm_pytorch.py for PyTorch version")
    print("3. Make sure your CSV files are in the same directory")
    print("4. Run the model script to start training!")
    print("\nIf you encounter any issues, try:")
    print("- Updating pip: python -m pip install --upgrade pip")
    print("- Using a virtual environment: python -m venv venv")
    print("- Installing specific versions from requirements.txt")