# JSONSchema Linter

This linter helps detect and prevent a common issue with Go struct tags in this project. 

## The Problem

In Go struct tags using `jsonschema`, commas in the `description` field need to be escaped using `\\,` syntax. If commas aren't properly escaped, the description is silently truncated at the comma.

For example:

```go
// Problematic (description will be truncated at the first comma):
type Example struct {
    Field string `jsonschema:"description=This is a description, but it will be truncated here"`
}

// Correct (commas properly escaped):
type Example struct {
    Field string `jsonschema:"description=This is a description\\, and it will be fully included"`
}
```

## Usage

You can use this linter by running:

```shell
make lint-jsonschema
```

or directly:

```shell
go run ./cmd/linters/jsonschema --path .
```

### Auto-fixing issues

The linter can automatically fix unescaped commas in jsonschema descriptions by running:

```shell
make lint-jsonschema-fix
```

or directly:

```shell
go run ./cmd/linters/jsonschema --path . --fix
```

This will scan the codebase for unescaped commas and automatically escape them, then report what was fixed.

## Flags

- `--path`: Base directory to scan for Go files (default: ".")
- `--fix`: Automatically fix unescaped commas
- `--help`: Display help information

## Integration

This linter is integrated into the default `make lint` command, ensuring all PRs are checked for this issue.