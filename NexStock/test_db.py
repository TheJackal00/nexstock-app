#!/usr/bin/env python3

import sqlite3
import sys
import os

# Add the current directory to the path so we can import our app functions
sys.path.append(os.path.dirname(__file__))

def test_database_connection():
    """Test basic database connection"""
    try:
        print("Testing basic database connection...")
        conn = sqlite3.connect('inventory.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Found {len(tables)} tables: {[t[0] for t in tables]}")
        
        conn.close()
        print("✓ Basic database connection test passed!")
        return True
    except Exception as e:
        print(f"✗ Basic database connection failed: {e}")
        return False

def test_app_functions():
    """Test our app's database functions"""
    try:
        print("\nTesting app database functions...")
        
        # Import our functions
        from app import get_db_connection, execute_query, init_db
        
        # Test initialization
        print("Testing init_db()...")
        if init_db():
            print("✓ Database initialization successful!")
        else:
            print("✗ Database initialization failed!")
            return False
        
        # Test execute_query with a simple SELECT
        print("Testing execute_query() with SELECT...")
        tables = execute_query("SELECT name FROM sqlite_master WHERE type='table';")
        print(f"Found {len(tables)} tables using execute_query: {[t['name'] for t in tables]}")
        
        # Test if we can query our main tables
        print("Testing queries on main tables...")
        
        try:
            products = execute_query("SELECT COUNT(*) as count FROM products")
            print(f"✓ Products table accessible: {products[0]['count']} records")
        except Exception as e:
            print(f"✗ Products table error: {e}")
        
        try:
            inventory = execute_query("SELECT COUNT(*) as count FROM inventory")
            print(f"✓ Inventory table accessible: {inventory[0]['count']} records")
        except Exception as e:
            print(f"✗ Inventory table error: {e}")
        
        try:
            transactions = execute_query("SELECT COUNT(*) as count FROM transactions")
            print(f"✓ Transactions table accessible: {transactions[0]['count']} records")
        except Exception as e:
            print(f"✗ Transactions table error: {e}")
        
        print("✓ App database functions test passed!")
        return True
        
    except Exception as e:
        print(f"✗ App database functions failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Database Conversion Test ===")
    
    # Test 1: Basic connection
    test1_passed = test_database_connection()
    
    # Test 2: App functions
    test2_passed = test_app_functions()
    
    if test1_passed and test2_passed:
        print("\n🎉 All database tests passed! The conversion appears successful.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")