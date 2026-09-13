"""
Keeps a Streamlit Community Cloud app awake by actually opening it in a
real (headless) browser, so the JS bundle loads and a genuine WebSocket
session gets established with the Streamlit server.

A plain curl/requests GET does NOT achieve this — it only downloads the
static HTML shell and exits, without ever running the app's Python script.
Streamlit Cloud's sleep-timer only resets on real sessions like this one.

This script also handles the case where the app has ALREADY gone to
sleep: it detects the "Zzzz... this app has gone to sleep" wake screen
and clicks the "Yes, get this app back up!" button, since a sleeping
app can only be woken by that click — not by any kind of passive visit.

It does not attempt to sign in — it doesn't need to. Even the
"Sign in to continue" screen behind Google auth is real content served
by a live Python session, which is exactly what counts as activity.
"""

import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = "https://voice-of-good-hope.streamlit.app"
WAKE_BUTTON_TEXT = "Yes, get this app back up!"


def ping_app() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Visiting {APP_URL} ...")
        page.goto(APP_URL, wait_until="networkidle", timeout=60_000)

        # If the app was asleep, click the wake-up button.
        wake_button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False)
        if wake_button.count() > 0:
            print("App was asleep — clicking wake-up button...")
            wake_button.first.click()
            # Waking a Community Cloud app can take a little while.
            page.wait_for_timeout(15_000)
            page.wait_for_load_state("networkidle", timeout=90_000)
        else:
            print("App was already awake.")

        # A harmless click on the page body to make sure the session
        # registers as a real visit, then hold the session open briefly.
        page.mouse.click(10, 10)
        time.sleep(5)

        browser.close()
        print("Done.")


if __name__ == "__main__":
    try:
        ping_app()
    except Exception as e:
        print(f"Ping failed: {e}", file=sys.stderr)
        sys.exit(1)
