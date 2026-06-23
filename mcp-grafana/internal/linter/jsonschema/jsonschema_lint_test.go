package linter

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindUnescapedCommas(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "jsonschema-linter-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	// Create test files
	testFiles := map[string]string{
		"valid.go": `package test

// Valid has properly escaped commas
type Valid struct {
	Name string ` + "`json:\"name\" jsonschema:\"description=A valid field\\, with escaped comma\"`" + `
	Age  int    ` + "`json:\"age\" jsonschema:\"description=Another valid field\"`" + `
}
`,
		"invalid.go": `package test

// Invalid has unescaped commas
type Invalid struct {
	Name string ` + "`json:\"name\" jsonschema:\"description=An invalid field, with unescaped comma\"`" + `
	Age  int    ` + "`json:\"age\" jsonschema:\"description=Another valid field\"`" + `
}
`,
		"mixed.go": `package test

// Mixed has both valid and invalid fields
type Mixed struct {
	Valid   string ` + "`json:\"valid\" jsonschema:\"description=A valid field\\, with escaped comma\"`" + `
	Invalid string ` + "`json:\"invalid\" jsonschema:\"description=An invalid field, with unescaped comma\"`" + `
}
`,
	}

	for filename, content := range testFiles {
		filePath := filepath.Join(tmpDir, filename)
		if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
			t.Fatalf("Failed to write test file %s: %v", filename, err)
		}
	}

	// Run the linter
	linter := &JSONSchemaLinter{}
	err = linter.FindUnescapedCommas(tmpDir)
	if err != nil {
		t.Fatalf("Linter failed: %v", err)
	}

	// Check if we found the expected errors
	if len(linter.Errors) != 2 {
		t.Errorf("Expected 2 errors, got %d", len(linter.Errors))
	}

	// Check if the errors are in the expected files
	fileErrors := make(map[string]int)
	for _, e := range linter.Errors {
		fileName := filepath.Base(e.FilePath)
		fileErrors[fileName]++
	}

	if fileErrors["invalid.go"] != 1 {
		t.Errorf("Expected 1 error in invalid.go, got %d", fileErrors["invalid.go"])
	}

	if fileErrors["mixed.go"] != 1 {
		t.Errorf("Expected 1 error in mixed.go, got %d", fileErrors["mixed.go"])
	}

	if fileErrors["valid.go"] != 0 {
		t.Errorf("Expected 0 errors in valid.go, got %d", fileErrors["valid.go"])
	}
}

// TestEscapedQuotesWithComma tests if the regex correctly identifies unescaped commas
// in jsonschema tags that contain escaped quotes
func TestEscapedQuotesWithComma(t *testing.T) {
	testCases := []struct {
		tag         string
		shouldMatch bool
		description string
	}{
		{`jsonschema:"description=This has an unescaped, comma"`, true, "Simple unescaped comma"},
		{`jsonschema:"description=This has escaped quote \"followed by, comma"`, true, "Escaped quote then unescaped comma"},
		{`jsonschema:"description=This has escaped quote \", comma"`, true, "Escaped quote, comma with space"},
		{`jsonschema:"description=This has escaped quote \\\"and escaped\\, comma"`, false, "Properly escaped quote and comma"},
		{`jsonschema:"description=No comma here"`, false, "No comma at all"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			matches := tagPattern.FindStringSubmatch(tc.tag)
			hasMatch := len(matches) > 0
			if hasMatch != tc.shouldMatch {
				t.Fatalf("Test failed for %s: expected match=%v, got=%v\n", tc.description, tc.shouldMatch, hasMatch)
			}
		})
	}
}

func TestFixUnescapedCommas(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "jsonschema-linter-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	// Create a test file with unescaped commas
	invalidContent := `package test

// Invalid has unescaped commas
type Invalid struct {
	Name string ` + "`json:\"name\" jsonschema:\"description=An invalid field, with unescaped comma\"`" + `
	Age  int    ` + "`json:\"age\" jsonschema:\"description=Another field, also with unescaped comma\"`" + `
}
`

	// Expected content after fixing
	// Note: We need double backslashes in the actual file, so we use double escaped backslashes here
	expectedContent := `package test

// Invalid has unescaped commas
type Invalid struct {
	Name string ` + "`json:\"name\" jsonschema:\"description=An invalid field\\\\, with unescaped comma\"`" + `
	Age  int    ` + "`json:\"age\" jsonschema:\"description=Another field\\\\, also with unescaped comma\"`" + `
}
`

	filePath := filepath.Join(tmpDir, "invalid.go")
	if err := os.WriteFile(filePath, []byte(invalidContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	// Run the linter with fix mode enabled
	linter := &JSONSchemaLinter{FixMode: true}
	err = linter.FindUnescapedCommas(tmpDir)
	if err != nil {
		t.Fatalf("Linter failed: %v", err)
	}

	// Check if we found the expected errors
	if len(linter.Errors) != 2 {
		t.Errorf("Expected 2 errors, got %d", len(linter.Errors))
	}

	// Verify the file was fixed
	fixedContent, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("Failed to read fixed file: %v", err)
	}

	if string(fixedContent) != expectedContent {
		t.Errorf("File not fixed correctly.\nExpected:\n%s\n\nGot:\n%s", expectedContent, string(fixedContent))
	}

	// Verify the fixed field was correctly tracked
	if !linter.Fixed[filePath] {
		t.Errorf("Fixed file not tracked in linter.Fixed")
	}
}
