"""
Quick setup verification script
"""
import sys

def check_installation():
    try:
        import domynclaimalign
        print("✅ domynclaimalign package imported successfully")
        
        # Check main modules
        from domynclaimalign.main import compute_traces
        print("✅ Main modules accessible")
        
        from domynclaimalign.utils import model_utils
        print("✅ Utility modules accessible")
        
        print("🎉 Installation verified successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)