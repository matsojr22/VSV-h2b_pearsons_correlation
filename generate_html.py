#!/usr/bin/env python3
"""Generate HTML from markdown files using unified style template."""
import argparse
import os
import sys
import markdown
from pathlib import Path

# Add scripts directory to path to import html_template
sys.path.insert(0, str(Path(__file__).parent))
from html_template import generate_html


def main():
    parser = argparse.ArgumentParser(description='Convert markdown to styled HTML')
    parser.add_argument('--input', '-i', required=True,
                        help='Input markdown file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output HTML file (default: same name as input with .html)')
    parser.add_argument('--title', '-t', default=None,
                        help='HTML page title (default: derived from filename)')
    args = parser.parse_args()

    # Read markdown
    with open(args.input, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite', 'toc', 'nl2br'])
    html_body = md.convert(md_content)
    
    # Fix links: replace .md with .html in href attributes
    import re
    html_body = re.sub(r'href="([^"]+)\.md"', r'href="\1.html"', html_body)
    html_body = re.sub(r'href="([^"]+)\.md#', r'href="\1.html#', html_body)

    # Determine title
    title = args.title
    if title is None:
        # Extract title from first H1 in markdown or use filename
        lines = md_content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('# '):
                title = line[2:].strip()
                break
        if title is None:
            title = Path(args.input).stem.replace('_', ' ').title()

    # Apply template
    html_output = generate_html(html_body, title=title)

    # Write output
    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.input).with_suffix('.html'))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f'HTML generated: {output_path}')


if __name__ == '__main__':
    main()
