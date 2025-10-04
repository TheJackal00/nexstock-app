#!/usr/bin/env python3

import requests
import json

def test_optimization_endpoint():
    print("=== Testing Optimization Endpoint ===")
    
    base_url = "http://127.0.0.1:5000"
    
    try:
        # Test GET request to /optimize
        print("1. Testing GET /optimize...")
        response = requests.get(f"{base_url}/optimize", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ GET request successful")
            # Check if the page loads without immediate redirect
            if "simulate" in response.text.lower() and "run a simulation first" in response.text.lower():
                print("   ⚠️  Page shows 'run a simulation first' message")
            else:
                print("   ✓ Page loads correctly, no error message detected")
        else:
            print(f"   ✗ GET request failed: {response.text[:200]}")
            
        # Test POST request to /optimize (trigger optimization)
        print("\n2. Testing POST /optimize...")
        response = requests.post(f"{base_url}/optimize", timeout=60)  # Longer timeout for optimization
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ POST request successful")
            if "error" in response.text.lower():
                print("   ⚠️  Response contains error message")
                # Look for error messages in the response
                if "no simulation data" in response.text.lower():
                    print("   Error: No simulation data found")
                elif "no optimization results" in response.text.lower():
                    print("   Error: No optimization results generated") 
            else:
                print("   ✓ Optimization appears to have completed successfully")
        else:
            print(f"   ✗ POST request failed: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError:
        print("   ✗ Could not connect to Flask server. Make sure it's running on port 5000.")
    except requests.exceptions.Timeout:
        print("   ✗ Request timed out. Optimization might be taking too long.")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

if __name__ == "__main__":
    test_optimization_endpoint()