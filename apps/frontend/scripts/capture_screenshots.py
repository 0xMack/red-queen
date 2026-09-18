"""Regenerates the real screenshots in public/screenshots/ (game and Learn chapter covers) from the
running app -- they're captured, not drawn, so re-run this after visual changes.

Needs `pnpm dev` (port 3000) and apis/backend (port 8000) running with a trained Snake run. From
the repo root:

    uv run --with playwright python apps/frontend/scripts/capture_screenshots.py <snake_run_id>

(`uv run --with playwright python -m playwright install chromium` once if no browser is installed.)
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

BASE = "http://localhost:3000"
OUT = Path(__file__).resolve().parent.parent / "public" / "screenshots"
# Hide the Nuxt devtools badge that `pnpm dev` injects.
HIDE_DEVTOOLS = (
    "document.addEventListener('DOMContentLoaded', () => {"
    " const s = document.createElement('style');"
    " s.textContent = '#nuxt-devtools-container { display: none !important }';"
    " document.head.appendChild(s) })"
)


async def main(run_id: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        await ctx.add_init_script(HIDE_DEVTOOLS)

        # The champion mid-game: wait until it has eaten a few times.
        page = await ctx.new_page()
        await page.goto(f"{BASE}/watch/{run_id}", wait_until="networkidle")
        board = page.locator("svg.rounded-xl").first
        await board.wait_for(timeout=60_000)
        for _ in range(120):
            score = await page.locator("text=🍎").first.inner_text()
            if int(score.split()[0]) >= 6:
                break
            await page.wait_for_timeout(250)

        # Bare board for the game card: no overlay chips, and no mid-glide segments.
        await page.add_style_tag(content="[data-board-overlay] { visibility: hidden } svg g { transition: none !important }")
        await page.wait_for_timeout(60)
        await board.screenshot(path=OUT / "snake.png")
        await page.add_style_tag(content="[data-board-overlay] { visibility: visible }")
        await page.locator("div.card").first.screenshot(path=OUT / "snake-watch.png")

        # The run detail page, top of the fold.
        page = await ctx.new_page()
        await page.set_viewport_size({"width": 1600, "height": 1000})
        await page.goto(f"{BASE}/runs/{run_id}", wait_until="networkidle")
        await page.wait_for_timeout(9_000)  # Pyodide cold start + a few ticks
        await page.screenshot(path=OUT / "run-detail.png")

        await browser.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1]))
