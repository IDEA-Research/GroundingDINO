package linter

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// JSONSchemaLinter checks for unescaped commas in jsonschema struct tags
type JSONSchemaLinter struct {
	FilePaths []string
	Errors    []JSONSchemaError
	FixMode   bool
	Fixed     map[string]bool
}

// JSONSchemaError represents a linting error with file position details
type JSONSchemaError struct {
	FilePath string
	Line     int
	Column   int
	Offset   int // Byte offset in the file
	Struct   string
	Field    string
	Tag      string
	FixedTag string
}

// tagPattern matches jsonschema tags with description containing unescaped commas
// It captures:
// 1. The jsonschema tag
// 2. Parts of the description containing unescaped commas
// The pattern correctly handles:
// - Simple unescaped comma: "description=Something, with comma"
// - Escaped quote followed by unescaped comma: "description=With \"quote, and comma"
// - But not match escaped comma: "description=With escaped\, comma"
var tagPattern = regexp.MustCompile(`jsonschema:"([^"]*)description=(.*?[^\\],)([^"]*)"`)

// FindUnescapedCommas scans Go files for jsonschema struct tags with unescaped commas in descriptions
func (l *JSONSchemaLinter) FindUnescapedCommas(baseDir string) error {
	// Reset errors
	l.Errors = nil
	if l.FixMode {
		l.Fixed = make(map[string]bool)
	}

	// Walk through the directory and find Go files
	err := filepath.Walk(baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip non-Go files
		if !info.IsDir() && strings.HasSuffix(path, ".go") {
			l.FilePaths = append(l.FilePaths, path)
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("error walking directory: %v", err)
	}

	// Parse all Go files and check for the unescaped commas
	for _, path := range l.FilePaths {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			return fmt.Errorf("error parsing file %s: %v", path, err)
		}

		fileErrors := []JSONSchemaError{}

		// Visit all struct types
		ast.Inspect(f, func(n ast.Node) bool {
			ts, ok := n.(*ast.TypeSpec)
			if !ok || ts.Type == nil {
				return true
			}

			st, ok := ts.Type.(*ast.StructType)
			if !ok {
				return true
			}

			structName := ts.Name.Name

			// Check each field of the struct
			for _, field := range st.Fields.List {
				if field.Tag == nil {
					continue
				}

				tag := field.Tag.Value

				// Check if the tag has a jsonschema description with unescaped comma
				matches := tagPattern.FindStringSubmatch(tag)
				if len(matches) > 0 {
					fieldName := ""
					if len(field.Names) > 0 {
						fieldName = field.Names[0].Name
					}

					// Generate the fixed tag by escaping the commas in the description
					fixedTag := tag
					if len(matches) > 2 {
						descWithUnescapedCommas := matches[2]
						// Escape all unescaped commas
						fixedDesc := escapeUnescapedCommas(descWithUnescapedCommas)
						// Replace the original description with the fixed one
						fixedTag = strings.Replace(tag, descWithUnescapedCommas, fixedDesc, 1)
					}

					pos := fset.Position(field.Tag.Pos())
					errorInfo := JSONSchemaError{
						FilePath: path,
						Line:     pos.Line,
						Column:   pos.Column,
						Offset:   pos.Offset,
						Struct:   structName,
						Field:    fieldName,
						Tag:      tag,
						FixedTag: fixedTag,
					}
					fileErrors = append(fileErrors, errorInfo)
				}
			}

			return true
		})

		// Add all errors for this file
		l.Errors = append(l.Errors, fileErrors...)

		// If in fix mode and we found errors, fix the file
		if l.FixMode && len(fileErrors) > 0 {
			err := l.fixFile(path, fileErrors)
			if err != nil {
				return fmt.Errorf("error fixing file %s: %v", path, err)
			}
			l.Fixed[path] = true
		}
	}

	return nil
}

// escapeUnescapedCommas escapes any unescaped commas in the description
func escapeUnescapedCommas(desc string) string {
	// Use regex to find all commas that are not preceded by a backslash
	r := regexp.MustCompile(`([^\\]),`)
	// Replace them with the same text but with an escaped comma
	return r.ReplaceAllString(desc, `$1\\,`)
}

// fixFile applies the fixes to a file
func (l *JSONSchemaLinter) fixFile(path string, errors []JSONSchemaError) error {
	// Read the file content
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("error reading file %s: %v", path, err)
	}

	// Convert to string for easier manipulation
	fileContent := string(content)

	// Sort errors by offset in reverse order to avoid offset changes
	sort.Slice(errors, func(i, j int) bool {
		return errors[i].Offset > errors[j].Offset
	})

	// Apply fixes
	for _, e := range errors {
		// Find the tag in the file content
		tagStart := strings.Index(fileContent[e.Offset:], e.Tag)
		if tagStart == -1 {
			continue
		}
		absOffset := e.Offset + tagStart

		// Replace the tag with the fixed version
		fixedContent := fileContent[:absOffset] + e.FixedTag + fileContent[absOffset+len(e.Tag):]
		fileContent = fixedContent
	}

	// Write back to the file
	err = os.WriteFile(path, []byte(fileContent), 0644)
	if err != nil {
		return fmt.Errorf("error writing file %s: %v", path, err)
	}

	return nil
}

// PrintErrors outputs all the found errors
func (l *JSONSchemaLinter) PrintErrors() {
	if len(l.Errors) == 0 {
		fmt.Println("No unescaped commas found in jsonschema descriptions.")
		return
	}

	if l.FixMode {
		fmt.Printf("Found and fixed %d unescaped commas in jsonschema descriptions:\n\n", len(l.Errors))
	} else {
		fmt.Printf("Found %d unescaped commas in jsonschema descriptions:\n\n", len(l.Errors))
	}

	for i, err := range l.Errors {
		relPath, _ := filepath.Rel(".", err.FilePath)
		fmt.Printf("%d. %s:%d:%d - Struct: %s, Field: %s\n",
			i+1, relPath, err.Line, err.Column, err.Struct, err.Field)
		fmt.Printf("   - %s\n", err.Tag)
		if l.FixMode {
			fmt.Printf("   - Fixed to: %s\n\n", err.FixedTag)
		} else {
			fmt.Printf("   - Commas in description must be escaped with \\\\,\n\n")
		}
	}

	if !l.FixMode {
		fmt.Println("Please escape all commas in jsonschema descriptions with \\\\, to prevent truncation.")
		fmt.Println("You can run with --fix to automatically fix these issues.")
	} else {
		fixedFileCount := len(l.Fixed)
		fmt.Printf("Fixed %d file(s).\n", fixedFileCount)
	}
}
