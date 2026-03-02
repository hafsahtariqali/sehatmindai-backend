"""
Quick script to test if Groq API key is set correctly.
Run this before starting your server to verify the API key is accessible.
"""

import os

def test_api_key():
    """Test if GROQ_API_KEY environment variable is set."""
    
    api_key = os.getenv("GROQ_API_KEY")
    
    print("=" * 60)
    print("Groq API Key Test")
    print("=" * 60)
    
    if api_key:
        # Show first 10 and last 4 characters for security
        masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        print(f"[OK] API Key is SET")
        print(f"  Key preview: {masked_key}")
        print(f"  Full length: {len(api_key)} characters")
        
        return True
    else:
        print("[ERROR] API Key is NOT SET")
        print("\nTo set it:")
        print("  Command Prompt: set GROQ_API_KEY=your-key-here")
        print("  PowerShell:     $env:GROQ_API_KEY=\"your-key-here\"")
        print("\nIMPORTANT: Set it in the SAME command prompt window where you run the server!")
        print("\nGet your API key from: https://console.groq.com/keys")
        return False

if __name__ == "__main__":
    test_api_key()
    print("\n" + "=" * 60)


