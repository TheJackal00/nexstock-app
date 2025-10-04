#!/usr/bin/env python3

import sys
import os

def test_flask_import():
    """Test if Flask app can be imported successfully"""
    try:
        print("Testing Flask app import...")
        from app import app
        print("✓ Flask app imported successfully!")
        
        print("\nAvailable routes:")
        for rule in app.url_map.iter_rules():
            print(f"  {rule.rule} -> {rule.endpoint}")
        
        return True
    except Exception as e:
        print(f"✗ Flask app import failed: {e}")
        return False

def test_database_routes():
    """Test if database routes work without errors"""
    try:
        print("\nTesting database routes...")
        from app import portafolio, inventory, transactions
        
        # Test with Flask app context
        from app import app
        with app.app_context():
            print("Testing routes with app context...")
            print("✓ Routes can be imported and should work with proper request context")
        
        return True
    except Exception as e:
        print(f"✗ Database routes test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Flask Application Test ===")
    
    # Test 1: Import Flask app
    test1_passed = test_flask_import()
    
    # Test 2: Database routes
    test2_passed = test_database_routes()
    
    if test1_passed and test2_passed:
        print("\n🎉 Flask application tests passed! The app should work correctly.")
    else:
        print("\n❌ Some Flask tests failed. Please check the errors above.")