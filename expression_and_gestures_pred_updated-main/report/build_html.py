"""Render report.md and slides.md into styled HTML files.

Open the produced HTML in a browser and use Ctrl+P → Save as PDF to get a PDF.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


REPORT_CSS = """
:root {
  --bg: #FAFAF7;
  --ink: #1A1A1A;
  --muted: #666;
  --accent: #2F5D50;
  --hair: #E5E3DC;
}
* { box-sizing: border-box; }
body {
  font-family: Georgia, 'Source Serif Pro', 'Fraunces', serif;
  color: var(--ink);
  background: var(--bg);
  max-width: 780px;
  margin: 2.5rem auto;
  padding: 0 1.5rem 4rem;
  line-height: 1.65;
  font-size: 11pt;
}
h1, h2, h3, h4 { font-weight: 500; letter-spacing: -0.01em; }
h1 { font-size: 1.9rem; border-bottom: 1px solid var(--hair); padding-bottom: 0.4rem; margin-top: 0; }
h2 { font-size: 1.3rem; margin-top: 2.2rem; color: var(--accent); }
h3 { font-size: 1.05rem; margin-top: 1.5rem; }
p { margin: 0.6rem 0 0.8rem; }
a { color: var(--accent); }
code, pre { font-family: 'SF Mono', Menlo, monospace; font-size: 9.5pt; }
code { background: #F2F0E9; padding: 0.1rem 0.3rem; border-radius: 2px; }
pre {
  background: #F2F0E9;
  border: 1px solid var(--hair);
  padding: 0.8rem 1rem;
  border-radius: 2px;
  overflow-x: auto;
}
pre code { background: transparent; padding: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
  font-size: 10pt;
}
th, td {
  border: 1px solid var(--hair);
  padding: 0.4rem 0.7rem;
  text-align: left;
}
th { background: #F2F0E9; font-weight: 500; letter-spacing: 0.02em; }
tr:nth-child(even) td { background: #FBFAF5; }
img { max-width: 100%; height: auto; margin: 0.6rem 0; display: block; }
blockquote {
  border-left: 3px solid var(--accent);
  margin: 1rem 0;
  padding: 0.1rem 1rem;
  color: var(--muted);
  background: #F8F6EF;
}
hr { border: none; border-top: 1px solid var(--hair); margin: 1.8rem 0; }
.meta { color: var(--muted); font-size: 0.9rem; margin-top: -0.4rem; }
.authors { color: var(--muted); font-size: 0.95rem; }
@media print {
  body { margin: 1rem auto; }
  h2 { page-break-before: auto; }
  img { page-break-inside: avoid; }
}
"""


SLIDES_CSS = """
:root {
  --bg: #FAFAF7;
  --ink: #1A1A1A;
  --muted: #666;
  --accent: #2F5D50;
  --hair: #E5E3DC;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #E5E3DC; }
body {
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--ink);
}
.slide {
  background: var(--bg);
  width: 1280px;
  height: 720px;
  margin: 2rem auto;
  padding: 3rem 3.5rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  page-break-after: always;
}
.slide h1, .slide h2 {
  font-family: Georgia, 'Fraunces', serif;
  font-weight: 500;
  letter-spacing: -0.01em;
  margin: 0 0 1rem;
}
.slide h1 { font-size: 2.2rem; color: var(--accent); }
.slide h2 { font-size: 1.5rem; }
.slide h3 { font-size: 1.1rem; font-weight: 500; margin-top: 1.2rem; }
.slide p { font-size: 1rem; line-height: 1.55; margin: 0.4rem 0; }
.slide ul, .slide ol { font-size: 1rem; line-height: 1.6; }
.slide li { margin: 0.3rem 0; }
.slide code { font-family: 'SF Mono', Menlo, monospace; background: #F2F0E9; padding: 0.05rem 0.25rem; border-radius: 2px; font-size: 0.85rem; }
.slide pre { background: #F2F0E9; border: 1px solid var(--hair); padding: 0.8rem; font-size: 0.85rem; overflow: auto; }
.slide table { border-collapse: collapse; font-size: 0.95rem; margin: 0.6rem 0; }
.slide th, .slide td { border: 1px solid var(--hair); padding: 0.3rem 0.7rem; text-align: left; }
.slide th { background: #F2F0E9; font-weight: 500; }
.slide img { max-height: 350px; display: block; margin: 0.4rem auto; }
.slide blockquote { border-left: 3px solid var(--accent); padding: 0.2rem 0.8rem; color: var(--muted); margin: 0.6rem 0; }
.slide hr { display: none; }
.slide-number {
  position: absolute;
  bottom: 1rem;
  right: 1.5rem;
  font-size: 0.8rem;
  color: var(--muted);
  letter-spacing: 0.1em;
}
@media print {
  html, body { background: white; }
  .slide { margin: 0; box-shadow: none; }
  @page { size: 1280px 720px; margin: 0; }
}
"""


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end].strip()
            for line in block.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip().strip("'\"")
            text = text[end + 4 :].lstrip("\n")
    return meta, text


def _image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _inline_images(html: str, base: Path) -> str:
    def repl(m):
        src = m.group(1)
        if src.startswith(("http", "data:")):
            return m.group(0)
        resolved = (base / src).resolve()
        uri = _image_to_data_uri(resolved)
        return m.group(0).replace(src, uri) if uri else m.group(0)

    return re.sub(r'src="([^"]+)"', repl, html)


def _clean_pandoc_attrs(text: str) -> str:
    return re.sub(r"\{[^}]*\}", "", text)


def _render_report():
    src = (HERE / "report.md").read_text(encoding="utf-8")
    meta, body = _strip_frontmatter(src)
    body = _clean_pandoc_attrs(body)
    html = markdown.markdown(body, extensions=["tables", "fenced_code"])
    html = _inline_images(html, HERE)

    title = meta.get("title", "Report")
    authors = "<br>".join(
        a.strip("- ").strip()
        for a in meta.get("author", "").split(",")
        if a.strip()
    )

    page = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>{REPORT_CSS}</style>
</head><body>
<h1>{title}</h1>
<p class="meta">{meta.get("subtitle", "")}</p>
<p class="authors">{authors}</p>
<p class="meta">{meta.get("course", "")} · {meta.get("date", "")}</p>
<hr>
{html}
</body></html>"""

    out = HERE / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"  report -> {out}")


def _render_slides():
    src = (HERE / "slides.md").read_text(encoding="utf-8")
    _, body = _strip_frontmatter(src)
    body = _clean_pandoc_attrs(body)

    chunks = [c.strip() for c in re.split(r"\n---\n", body) if c.strip()]
    slides_html = []
    for i, chunk in enumerate(chunks, 1):
        inner = markdown.markdown(chunk, extensions=["tables", "fenced_code"])
        inner = _inline_images(inner, HERE)
        inner = re.sub(r"^<h1>(\d+\s*·\s*[^<]+)</h1>", r'<h2>\1</h2>', inner)
        slides_html.append(
            f'<section class="slide">{inner}<div class="slide-number">{i:02d}</div></section>'
        )

    page = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Slides</title>
<style>{SLIDES_CSS}</style>
</head><body>
{"".join(slides_html)}
</body></html>"""

    out = HERE / "slides.html"
    out.write_text(page, encoding="utf-8")
    print(f"  slides -> {out}  ({len(slides_html)} slides)")


def main():
    print("rendering HTML:")
    _render_report()
    _render_slides()
    print("\nopen the HTML files in a browser and use Ctrl+P -> Save as PDF.")


if __name__ == "__main__":
    main()
