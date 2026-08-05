package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	linter "github.com/grafana/mcp-grafana/internal/linter/jsonschema"
)

func main() {
	var (
		basePath string
		help     bool
		fix      bool
	)

	flag.StringVar(&basePath, "path", ".", "Base directory to scan for Go files")
	flag.BoolVar(&help, "help", false, "Show help message")
	flag.BoolVar(&fix, "fix", false, "Automatically fix unescaped commas")
	flag.Parse()

	if help {
		fmt.Println("jsonschema-linter - A tool to find unescaped commas in jsonschema struct tags")
		fmt.Println("\nUsage:")
		flag.PrintDefaults()
		os.Exit(0)
	}

	// Resolve to absolute path
	absPath, err := filepath.Abs(basePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error resolving path: %v\n", err)
		os.Exit(1)
	}

	// Initialize linter
	jsonLinter := &linter.JSONSchemaLinter{
		FixMode: fix,
	}

	// Find unescaped commas
	err = jsonLinter.FindUnescapedCommas(absPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error scanning files: %v\n", err)
		os.Exit(1)
	}

	// Print errors
	jsonLinter.PrintErrors()

	// Exit with error code if issues were found
	if len(jsonLinter.Errors) > 0 {
		os.Exit(1)
	}
}
