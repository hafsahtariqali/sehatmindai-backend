"""
Test script to check if server logs are working and if Groq is being called
Run this while your server is running to see what's happening
"""
import requests
import json

def test_chat_endpoint():
    """Test the chat endpoint and see what happens"""
    url = "http://localhost:8000/chat"
    
    payload = {
        "message": "Hello, I'm feeling sad today",
        "sender_id": "test_user_123"
    }
    
    print("=" * 60)
    print("Testing Chat Endpoint")
    print("=" * 60)
    print(f"Sending POST request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nSending request...")
    print("(Check your server logs to see what's happening)")
    print("=" * 60)
    
    try:
        response = requests.post(url, json=payload, timeout=25)
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse received:")
            print(json.dumps(result, indent=2))
            
            if result.get("responses"):
                response_text = result["responses"][0].get("text", "")
                print(f"\nBot response: {response_text}")
                
                if "I'm here to listen" in response_text:
                    print("\n⚠ WARNING: Got fallback response - Groq API may not have been called")
                else:
                    print("\n✓ Got custom response - Groq API was likely called")
        else:
            print(f"\nError: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to server")
        print("Make sure your server is running on http://localhost:8000")
    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out after 25 seconds")
        print("This suggests the server is hanging or taking too long")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chat_endpoint()

