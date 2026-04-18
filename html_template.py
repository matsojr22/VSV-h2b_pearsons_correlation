"""Unified HTML template for MAPseq documentation."""

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        :root {{
            --primary: #1a365d;
            --secondary: #2c5282;
            --accent: #3182ce;
            --light-accent: #90cdf4;
            --bg: #ffffff;
            --code-bg: #1a202c;
            --code-text: #e2e8f0;
            --table-border: #cbd5e0;
            --table-header: #edf2f7;
            --table-alt: #f7fafc;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 2rem; line-height: 1.7; color: #333; background: var(--bg); }}
        h1 {{ color: var(--primary); border-bottom: 3px solid var(--accent); padding-bottom: 0.5rem; font-size: 2rem; }}
        h2 {{ color: var(--secondary); border-bottom: 2px solid var(--light-accent); padding-bottom: 0.3rem; margin-top: 2.5rem; font-size: 1.5rem; }}
        h3 {{ color: #2b6cb0; margin-top: 1.8rem; font-size: 1.2rem; }}
        h4 {{ color: #4a5568; margin-top: 1.5rem; font-size: 1.05rem; }}
        code {{ background: #edf2f7; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.9em; font-family: "SF Mono", Monaco, monospace; }}
        pre {{ background: var(--code-bg); color: var(--code-text); padding: 1.2rem; border-radius: 8px; overflow-x: auto; font-size: 0.85em; }}
        pre code {{ background: none; color: inherit; padding: 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1.5rem 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid var(--table-border); padding: 0.6rem 0.8rem; text-align: left; }}
        th {{ background: var(--table-header); font-weight: 600; color: var(--primary); }}
        tr:nth-child(even) {{ background: var(--table-alt); }}
        blockquote {{ border-left: 4px solid var(--accent); margin: 1.5rem 0; padding: 0.5rem 1rem; color: #4a5568; background: #f7fafc; border-radius: 0 8px 8px 0; }}
        hr {{ border: none; border-top: 2px solid #e2e8f0; margin: 2.5rem 0; }}
        .mermaid {{ background: #f7fafc; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; text-align: center; }}
        strong {{ color: var(--secondary); }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        ul, ol {{ margin: 1rem 0; padding-left: 2rem; }}
        li {{ margin: 0.3rem 0; }}
        .MathJax {{ font-size: 1.1em !important; }}
        .nav {{ background: #f7fafc; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }}
        .nav a {{ margin-right: 1rem; }}
        @media print {{ body {{ max-width: 100%; padding: 1rem; }} pre {{ white-space: pre-wrap; }} }}
    </style>
</head>
<body>
{body}
<script>
    mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    document.querySelectorAll('pre code').forEach(block => {{
        if (block.innerText.trim().startsWith('flowchart') || 
            block.innerText.trim().startsWith('graph') || 
            block.innerText.trim().startsWith('sequenceDiagram')) {{
            const div = document.createElement('div');
            div.className = 'mermaid';
            div.textContent = block.innerText;
            block.parentElement.replaceWith(div);
        }}
    }});
</script>
</body>
</html>'''


def generate_html(body_content, title="MAPseq Documentation"):
    """Generate complete HTML document with unified styling."""
    return HTML_TEMPLATE.format(title=title, body=body_content)
