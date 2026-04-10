import requests
import json
import time

# Use plain text for log output to avoid Windows console Unicode errors
API_URL = "http://localhost:3001/api/chat"

TEST_QUERIES = [
    {"msg": "Explain the Central Limit Theorem with a normal distribution graph", "mode": "single"},
    {"msg": "Perform a Pareto analysis on customer complaints", "mode": "multi"},
    {"msg": "What is the probability of 5 successes in a Poisson distribution?", "mode": "single"},
    {"msg": "Draw a pie chart of favorite fruits", "mode": "multi"},
    {"msg": "Compare three groups using ANOVA", "mode": "multi"}
]

def run_test():
    results = []
    print(">>> Starting Automated Triple-Sync Stress Test...")
    
    # Run 15 trials to hit the stochastic dispatcher multiple times
    for i in range(15):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        start_time = time.time()
        
        try:
            print(f"Trial {i+1}/15 | Query: {query['msg'][:30]}... | Mode: {query['mode']}")
            res = requests.post(API_URL, data={
                "message": query['msg'],
                "mode": query['mode'],
                "domain": "statistics"
            }, timeout=60)
            
            latency = round(time.time() - start_time, 2)
            
            if res.status_code == 200:
                data = res.json()
                reply = data.get('reply', '')
                model = data.get('model_used', 'unknown')
                
                # Validation
                has_explanation = "<explanation>" in reply
                has_chart = "<chart_params>" in reply
                
                status = "PASS" if (has_explanation or ("chart_params" in reply)) else "FAIL (No Tags)"
                
                results.append({
                    "trial": i+1,
                    "model": model,
                    "latency": latency,
                    "status": status,
                    "query": query['msg']
                })
                print(f"   [OK] {model} | {latency}s | {status}")
            else:
                print(f"   [ERR] HTTP {res.status_code}")
                results.append({"trial": i+1, "status": f"HTTP {res.status_code}", "query": query['msg']})
                
        except Exception as e:
            print(f"   [ERR] Error: {e}")
            results.append({"trial": i+1, "status": f"Error: {e}", "query": query['msg']})
            
        time.sleep(1) # Breathe
        
    # Generate Report
    report_path = "Triple_Sync_Stress_Test_Report.md"
    try:
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("# Triple-Sync Stress Test Report\n\n")
            f.write(f"**Timestamp:** {time.ctime()}\n\n")
            f.write("| Trial | Provider:Model | Latency | Status | Query |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for r in results:
                f.write(f"| {r.get('trial')} | {r.get('model','N/A')} | {r.get('latency','N/A')}s | {r.get('status')} | {r.get('query')} |\n")
        print(f"\nTest Complete! Report saved to {report_path}")
    except Exception as ex:
        print(f"Error saving report: {ex}")

if __name__ == "__main__":
    run_test()
