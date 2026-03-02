"""
Direct test of Groq API to debug connection issues
"""
import os
import httpx
import asyncio

async def test_groq_api():
    """Test Groq API directly"""
    api_key = os.getenv("GROQ_API_KEY")
    
    print("=" * 60)
    print("Testing Groq API Directly")
    print("=" * 60)
    
    if not api_key:
        print("ERROR: GROQ_API_KEY not set!")
        return
    
    print(f"API Key found: {api_key[:10]}...{api_key[-4:]}")
    print(f"API Key length: {len(api_key)}")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    print(f"\nMaking request to: {url}")
    print(f"Model: llama-3.1-8b-instant")
    print(f"Timeout: 15 seconds")
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            print("\nSending request...")
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("\nSUCCESS! Response received:")
                print(result)
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print(f"\nGenerated text: {content}")
            else:
                print(f"\nERROR: {response.status_code}")
                print(f"Response text: {response.text}")
                
    except httpx.TimeoutException:
        print("\nERROR: Request timed out after 15 seconds")
        print("This could mean:")
        print("  - Network connectivity issues")
        print("  - Groq API is down")
        print("  - Firewall blocking the request")
    except httpx.RequestError as e:
        print(f"\nERROR: Request failed: {e}")
        print(f"Error type: {type(e)}")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_groq_api())
    print("\n" + "=" * 60)

