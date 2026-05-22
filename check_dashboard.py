import time
from playwright.sync_api import sync_playwright

def check_dashboard():
    print("Launching headless browser to inspect Streamlit dashboard...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Navigate to dashboard
        page.goto("http://localhost:8501")
        
        # Wait for the Streamlit app to load (Streamlit adds a 'stApp' div)
        print("Waiting for app to load...")
        try:
            page.wait_for_selector(".stApp", timeout=10000)
            # Give it an extra few seconds to render all charts
            time.sleep(3)
        except Exception as e:
            print("Timeout waiting for .stApp:", e)
            
        print("\n--- DASHBOARD TEXT CONTENT ---")
        # Extract text from the main body
        # Streamlit puts the main content in a div with data-testid="stAppViewBlockContainer"
        try:
            content = page.locator(".stApp").inner_text()
            print(content)
        except Exception as e:
            print("Could not extract main content:", e)
            
        print("\n--- SIDEBAR TEXT CONTENT ---")
        try:
            sidebar = page.locator("section[data-testid='stSidebar']").inner_text()
            print(sidebar)
        except Exception as e:
            print("Could not extract sidebar:", e)
            
        browser.close()

if __name__ == "__main__":
    check_dashboard()
