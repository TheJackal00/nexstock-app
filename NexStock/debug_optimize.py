#!/usr/bin/env python3

from app import execute_query

def test_optimization_data():
    print("=== Testing Optimization Data ===")
    
    # Check products
    products = execute_query('SELECT SKU FROM products')
    print(f"Available SKUs: {[p['SKU'] for p in products]}")
    
    # Check simulation data for each SKU
    for product in products:
        sku = product['SKU']
        sim_data = execute_query('SELECT COUNT(*) as count FROM Simulation WHERE SKU = ?', sku)
        print(f"SKU {sku}: {sim_data[0]['count']} simulation records")
        
        # Check iterations for this SKU
        if sim_data[0]['count'] > 0:
            iterations = execute_query('SELECT DISTINCT Iteration FROM Simulation WHERE SKU = ? ORDER BY Iteration', sku)
            print(f"  Available iterations: {[i['Iteration'] for i in iterations]}")
    
    # Check total simulation records
    total_sim = execute_query('SELECT COUNT(*) as count FROM Simulation')
    print(f"\nTotal simulation records: {total_sim[0]['count']}")
    
    # Check for any recent iterations
    recent_iterations = execute_query('SELECT DISTINCT Iteration FROM Simulation ORDER BY Iteration DESC LIMIT 5')
    print(f"Recent iterations: {[r['Iteration'] for r in recent_iterations]}")

if __name__ == "__main__":
    test_optimization_data()