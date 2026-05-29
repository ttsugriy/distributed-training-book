#!/usr/bin/env python3
"""Render a built MkDocs (print-site) page to a single PDF book via Chromium.

It serves the built ``site_pdf/`` directory over a local HTTP server (so that
MathJax, Mermaid and relative assets all load exactly as in a browser), opens
the combined ``/print_page/`` with headless Chromium, waits for math + diagrams
to finish rendering, then prints to a tagged PDF.  Chromium preserves in-page
anchor links as clickable internal jumps, and emits a heading outline, so you
can navigate the book.

Usage:
    python scripts/print_to_pdf.py --site-dir site_pdf --output build/book.pdf
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(directory: Path):
    """Serve *directory* over a throwaway local HTTP server."""
    port = _free_port()
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )

    class QuietServer(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    httpd = QuietServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _wait_for_render(page, timeout_ms: int) -> None:
    """Best-effort wait until MathJax + Mermaid have finished rendering."""
    # Let lazy resources settle.
    with contextlib.suppress(Exception):
        page.wait_for_load_state("networkidle", timeout=timeout_ms)

    # Force MathJax to typeset and wait for it to report completion.
    with contextlib.suppress(Exception):
        page.evaluate(
            """async () => {
                if (window.MathJax && window.MathJax.typesetPromise) {
                    await window.MathJax.typesetPromise();
                }
            }"""
        )

    # Wait until every Mermaid source block has been replaced by an <svg>.
    with contextlib.suppress(Exception):
        page.wait_for_function(
            """() => {
                const blocks = Array.from(
                    document.querySelectorAll('pre.mermaid, .mermaid')
                );
                if (blocks.length === 0) return true;
                return blocks.every(b => b.querySelector('svg'));
            }""",
            timeout=timeout_ms,
        )

    # Wait for web fonts so glyph metrics are final before pagination.
    with contextlib.suppress(Exception):
        page.evaluate("async () => { await document.fonts.ready; }")

    # Small settle margin for any final reflow.
    time.sleep(2.0)


def render_pdf(url: str, output: Path, timeout_ms: int) -> None:
    from playwright.sync_api import sync_playwright

    output.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 1696})

        print(f"  -> loading {url}", flush=True)
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        print("  -> waiting for MathJax / Mermaid / fonts ...", flush=True)
        _wait_for_render(page, timeout_ms)

        print(f"  -> printing PDF to {output}", flush=True)
        pdf_kwargs = dict(
            path=str(output),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "18mm", "bottom": "20mm", "left": "16mm", "right": "16mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                "<div style='width:100%;font-size:8px;color:#888;"
                "text-align:center;'>"
                "<span class='pageNumber'></span> / "
                "<span class='totalPages'></span></div>"
            ),
        )
        # `tagged` / `outline` improve accessibility + bookmarks where supported.
        try:
            page.pdf(tagged=True, outline=True, **pdf_kwargs)
        except TypeError:
            page.pdf(**pdf_kwargs)

        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", default="site_pdf", help="Built site directory.")
    parser.add_argument(
        "--output", default="build/distributed-training-book.pdf", help="Output PDF."
    )
    parser.add_argument(
        "--page-path",
        default="print_page/index.html",
        help="Path (within site dir) of the combined print page.",
    )
    parser.add_argument(
        "--timeout", type=int, default=180_000, help="Per-step timeout in ms."
    )
    args = parser.parse_args()

    site_dir = Path(args.site_dir).resolve()
    page_file = site_dir / args.page_path
    if not page_file.exists():
        print(
            f"error: {page_file} not found. Run the mkdocs build first.",
            file=sys.stderr,
        )
        return 1

    output = Path(args.output).resolve()
    with serve(site_dir) as base_url:
        render_pdf(f"{base_url}/{args.page_path}", output, args.timeout)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\nDone: {output}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
