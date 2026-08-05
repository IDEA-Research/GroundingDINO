# Build stage
FROM golang:1.24-bullseye@sha256:2cdc80dc25edcb96ada1654f73092f2928045d037581fa4aa7c40d18af7dd85a AS builder

# Set the working directory
WORKDIR /app

# Copy go.mod and go.sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN go build -o mcp-grafana ./cmd/mcp-grafana

# Final stage
FROM debian:bullseye-slim@sha256:c5f48c942c667e70d7e64b124cfc939c25a4a43207c0d14b45844d762dc1d50f

LABEL io.modelcontextprotocol.server.name="io.github.grafana/mcp-grafana"

# Install ca-certificates for HTTPS requests
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -r -u 1000 -m mcp-grafana

# Set the working directory
WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder --chown=1000:1000 /app/mcp-grafana /app/

# Use the non-root user
USER mcp-grafana

# Expose the port the app runs on
EXPOSE 8000

# Run the application
ENTRYPOINT ["/app/mcp-grafana", "--transport", "sse", "--address", "0.0.0.0:8000"]
