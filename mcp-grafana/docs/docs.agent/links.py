#!/usr/bin/env python3

"""
Check Markdown links in documentation.

Usage: ./links.py <DOCS_SOURCE_PATH>
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Union

# Global list to collect error messages
errors: List[str] = []

def error(src: Union[str, Path], link: str, msg: str) -> None:
	"""
	Record an error message for a link in a source file.

	Args:
		src: Source file path where the error occurred.
		link: Problematic link string.
		msg: Descriptive error message.
	"""
	errors.append(f"ERROR: {src} - {msg}: {link}")

def main() -> None:
	"""
	Main execution function.

	Parse command line arguments, scan Markdown files for broken or malformed links,
	and report errors. Exit with status code 1 if errors exist.
	"""
	if len(sys.argv) < 2:
		sys.exit(f"Usage: {sys.argv[0]} <DOCS_SOURCE_PATH>")

	docs_dir = Path(sys.argv[1]).resolve()

	for path in docs_dir.rglob("*.md"):
		try:
			content = path.read_text(encoding="utf-8")
		except Exception as e:
			errors.append(f"ERROR: Could not read {path}: {e}")
			continue

		src = path.relative_to(docs_dir)

		# Find all Markdown links: [text](url)
		for link in re.findall(r"\[.*?\]\((.*?)\)", content):
			if not link.strip():
				continue

			# Skip external links, mailto links, and internal anchors
			if link.startswith(("http", "https", "mailto:", "#")):
				continue

			# Remove anchor from link to check file existence
			link = link.split("#")[0]

			# Remove optional title from link
			link = link.split()[0]

			# Grafana documentation links shouldn't include file extension
			if link.endswith(".md"):
				error(src, link, "Link has .md extension and should end in /")
				continue

			# Links should be relative, not absolute paths starting with /docs/
			if link.startswith("/docs/"):
				error(src, link, "Link is absolute and should be relative")
				continue

			# Determine base path for resolving relative links
			# If file is '_index.md', the base is parent directory
			# For other files, the base includes file stem
			base = path.parent if path.name == "_index.md" else path.parent / path.stem
			target = Path(os.path.normpath(base / link))

			# Check if target exists as .md file or directory with _index.md
			if not (target.with_suffix(".md").is_file() or (target / "_index.md").is_file()):
				error(src, link, "Target does not exist")

	if errors:
		print('\n'.join(errors))
		sys.exit(1)

	print("✓ All links are formatted correctly and all targets exist")

if __name__ == "__main__":
	main()
