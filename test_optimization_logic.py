#!/usr/bin/env python3

from app import execute_query, optimize_purchase_schedule

def test_optimization_logic():
    print("=== Testing Optimization Logic Directly ===")
    
    try:
        # Get product info
        products = execute_query("SELECT SKU, NAME, MARGIN, COST FROM products")
        product_info = {prod['SKU']: prod for prod in products}
        all_skus = [prod['SKU'] for prod in products]
        
        print(f"Found products: {[(p['SKU'], p['NAME']) for p in products]}")
        
        # Test optimization for the first SKU and first iteration
        sku = all_skus[0]
        iteration = 1
        
        print(f"\nTesting optimization for SKU {sku}, iteration {iteration}...")
        
        # Get simulation data for this SKU and iteration
        sim_rows = execute_query(
            "SELECT * FROM Simulation WHERE SKU = ? AND Iteration = ? ORDER BY Count",
            sku, iteration
        )
        
        if not sim_rows:
            print(f"✗ No simulation data found for SKU {sku}, iteration {iteration}")
            return False
            
        print(f"✓ Found {len(sim_rows)} simulation records")
        
        # Extract data for optimization
        demands = [int(row['DEMAND']) for row in sim_rows]
        lead_times = [int(row['LEAD_TIME']) for row in sim_rows]
        initial_stock = float(sim_rows[0]['STOCK']) + int(sim_rows[0]['DEMAND']) if sim_rows else 0
        
        unit_cost = float(product_info[sku]['COST']) if product_info[sku]['COST'] else 10
        margin = float(product_info[sku]['MARGIN']) if product_info[sku]['MARGIN'] else 0
        
        print(f"  Demands: {demands[:5]}... (first 5)")
        print(f"  Lead times: {lead_times[:5]}... (first 5)")
        print(f"  Initial stock: {initial_stock}")
        print(f"  Unit cost: {unit_cost}")
        print(f"  Margin: {margin}")
        
        # Run optimization
        print("\nRunning optimization...")
        result = optimize_purchase_schedule(
            sku=sku,
            demands=demands,
            lead_times=lead_times,
            initial_stock=initial_stock,
            unit_cost=unit_cost,
            margin=margin,
            shipping_cost=100,
            holding_cost_rate=0.01,
            stockout_penalty_multiplier=2.0,
            simulation_days=len(demands)
        )
        
        print("✓ Optimization completed successfully!")
        print(f"  Total cost: ${result['optimized_strategy']['total_cost']:.2f}")
        print(f"  Service level: {result['optimized_strategy']['service_level']:.1f}%")
        print(f"  Total orders: {result['optimized_strategy']['total_orders']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimization_logic()
    if success:
        print("\n🎉 Optimization logic test passed!")
        print("The issue might be in the Flask route handling, not the optimization logic itself.")
    else:
        print("\n❌ Optimization logic test failed.")
        print("There's an issue with the optimization algorithm or data processing.")