from playwright.sync_api import sync_playwright, Playwright, BrowserType
from bedrock_agentcore.tools.browser_client import browser_session
from browser_viewer import BrowserViewerServer
import time
from rich.console import Console

console = Console()

def run(playwright: Playwright):
    # Create the browser session and keep it alive
    with browser_session('us-west-2') as client:
        ws_url, headers = client.generate_ws_headers()

        # Start viewer server
        viewer = BrowserViewerServer(client, port=8005)
        viewer_url = viewer.start(open_browser=True)

        # Connect using headers
        chromium: BrowserType = playwright.chromium
        browser = chromium.connect_over_cdp(
            ws_url,
            headers=headers
        )

        context = browser.contexts[0]
        page = context.pages[0]

        try:
            while True:
                page.goto("https://google.com/")
                console.print(page.title())
                # Keep running
                while True:
                    time.sleep(120)
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Shutting down...[/yellow]")
            if 'client' in locals():
                client.stop()
                console.print("✅ Browser session terminated")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()

with sync_playwright() as playwright:
    run(playwright)