"""
Thin wrapper around a real (headless) Playwright browser for the Research
agent, reused from the SOC-2026 browser agent project (script5_agent.py).

Why this exists instead of just using `requests`: some careers/about pages
render content with JavaScript, so a plain HTTP GET returns an empty shell.
A real browser handles that. The thread-pinning pattern below is carried
over as-is from script5_agent.py — I hit the exact "Cannot switch to a
different thread" error there because Playwright's sync API locks a Page to
whichever thread created it, and LangGraph (or here, just calling this from
different contexts) doesn't guarantee the same thread every time. Routing
every real Playwright call through one dedicated worker thread fixes it.
"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout

_pw_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright-worker")


def _run_on_pw_thread(fn, *args, **kwargs):
    return _pw_executor.submit(fn, *args, **kwargs).result()


_playwright_instance = None
_browser = None
_context: BrowserContext = None
_page: Page = None


def _get_page_impl() -> Page:
    global _playwright_instance, _browser, _context, _page
    if _page is None:
        _playwright_instance = sync_playwright().start()
        # headless=True here (unlike the SOC-2026 agent) since this just
        # needs to read a page, not demo actions visibly to a person.
        _browser = _playwright_instance.chromium.launch(headless=True)
        _context = _browser.new_context()
        _page = _context.new_page()
    return _page


def _fetch_page_text_impl(url: str, timeout_ms: int = 15000) -> str:
    page = _get_page_impl()
    page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
    return page.inner_text("body")


def fetch_page_text(url: str) -> str:
    """Navigate to a URL with a real headless browser and return the visible
    page text. Raises on navigation/timeout errors — caller decides the
    fallback."""
    try:
        return _run_on_pw_thread(_fetch_page_text_impl, url)
    except PlaywrightTimeout as e:
        raise RuntimeError(f"Timed out loading {url}: {e}")


def _close_impl():
    global _playwright_instance, _browser, _context, _page
    try:
        if _page and not _page.is_closed():
            _page.close()
        if _context:
            _context.close()
        if _browser:
            _browser.close()
        if _playwright_instance:
            _playwright_instance.stop()
    except Exception:
        pass
    finally:
        _page = None
        _context = None
        _browser = None
        _playwright_instance = None


def close_browser():
    if _page is None and _browser is None:
        return  # never launched, nothing to clean up
    try:
        _run_on_pw_thread(_close_impl)
    except RuntimeError:
        pass  # interpreter already shutting down; executor can't take new work


atexit.register(close_browser)
