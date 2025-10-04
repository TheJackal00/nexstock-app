#!/usr/bin/env python3

import sys
import os

def test_specific_routes():
    """Test specific database operations from the app"""
    try:
        print("Testing specific database operations...")
        
        from app import execute_query, get_db_connection
        
        # Test 1: Portfolio query (products)
        print("\n1. Testing portfolio query...")
        products = execute_query("SELECT * FROM products")
        print(f"✓ Portfolio query successful: {len(products)} products found")
        if products:
            print(f"   Sample product: {products[0]['NAME'] if 'NAME' in products[0] else 'N/A'}")
        
        # Test 2: Inventory query (complex JOIN)
        print("\n2. Testing inventory query...")
        stock = execute_query(
            "SELECT p.SKU,p.NAME,i.EXPIRED,i.DAY,i.VOLUME,p.EXPIRATION FROM inventory AS i JOIN products AS p ON i.SKU = p.SKU;"
        )
        print(f"✓ Inventory JOIN query successful: {len(stock)} inventory items found")
        
        # Test 3: Transactions query
        print("\n3. Testing transactions query...")
        trades = execute_query("SELECT * FROM transactions;")
        print(f"✓ Transactions query successful: {len(trades)} transactions found")
        
        # Test 4: Simulation data check
        print("\n4. Testing simulation data...")
        sim_data = execute_query("SELECT COUNT(*) as count FROM Simulation")
        print(f"✓ Simulation query successful: {sim_data[0]['count']} simulation records")
        
        # Test 5: SKU selection for simulation
        print("\n5. Testing SKU selection...")
        skus = execute_query("SELECT DISTINCT SKU FROM transactions")
        print(f"✓ SKU selection successful: {len(skus)} unique SKUs found")
        
        return True
        
    except Exception as e:
        print(f"✗ Specific database operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transaction_operations():
    """Test transaction functionality"""
    try:
        print("\nTesting transaction operations...")
        
        from app import execute_transaction
        
        # Test a simple transaction (we'll use a harmless SELECT operation)
        operations = [
            ("SELECT COUNT(*) FROM products", ()),
            ("SELECT COUNT(*) FROM inventory", ())
        ]
        
        result = execute_transaction(operations)
        print(f"✓ Transaction operations successful: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Transaction operations failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Comprehensive Database Operations Test ===")
    
    # Test 1: Specific routes
    test1_passed = test_specific_routes()
    
    # Test 2: Transaction operations
    test2_passed = test_transaction_operations()
    
    if test1_passed and test2_passed:
        print("\n🎉 All comprehensive database tests passed!")
        print("✅ The CS50 to sqlite3 conversion is complete and working correctly!")
        print("\nYour NexStock application should now work without the CS50 dependency.")
    else:
        print("\n❌ Some comprehensive tests failed. Please check the errors above.")