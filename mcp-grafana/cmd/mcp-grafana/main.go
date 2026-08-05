package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	mcpgrafana "github.com/grafana/mcp-grafana"
	"github.com/grafana/mcp-grafana/tools"
)

func maybeAddTools(s *server.MCPServer, tf func(*server.MCPServer), enabledTools []string, disable bool, category string) {
	if !slices.Contains(enabledTools, category) {
		slog.Debug("Not enabling tools", "category", category)
		return
	}
	if disable {
		slog.Info("Disabling tools", "category", category)
		return
	}
	slog.Debug("Enabling tools", "category", category)
	tf(s)
}

// disabledTools indicates whether each category of tools should be disabled.
type disabledTools struct {
	enabledTools string

	search, datasource, incident,
	prometheus, loki, alerting,
	dashboard, folder, oncall, asserts, sift, admin,
	pyroscope, navigation, proxied, annotations, write bool
}

// Configuration for the Grafana client.
type grafanaConfig struct {
	// Whether to enable debug mode for the Grafana transport.
	debug bool

	// TLS configuration
	tlsCertFile   string
	tlsKeyFile    string
	tlsCAFile     string
	tlsSkipVerify bool
}

func (dt *disabledTools) addFlags() {
	flag.StringVar(&dt.enabledTools, "enabled-tools", "search,datasource,incident,prometheus,loki,alerting,dashboard,folder,oncall,asserts,sift,pyroscope,navigation,proxied,annotations", "A comma separated list of tools enabled for this server. Can be overwritten entirely or by disabling specific components, e.g. --disable-search.")
	flag.BoolVar(&dt.search, "disable-search", false, "Disable search tools")
	flag.BoolVar(&dt.datasource, "disable-datasource", false, "Disable datasource tools")
	flag.BoolVar(&dt.incident, "disable-incident", false, "Disable incident tools")
	flag.BoolVar(&dt.prometheus, "disable-prometheus", false, "Disable prometheus tools")
	flag.BoolVar(&dt.loki, "disable-loki", false, "Disable loki tools")
	flag.BoolVar(&dt.alerting, "disable-alerting", false, "Disable alerting tools")
	flag.BoolVar(&dt.dashboard, "disable-dashboard", false, "Disable dashboard tools")
	flag.BoolVar(&dt.folder, "disable-folder", false, "Disable folder tools")
	flag.BoolVar(&dt.oncall, "disable-oncall", false, "Disable oncall tools")
	flag.BoolVar(&dt.asserts, "disable-asserts", false, "Disable asserts tools")
	flag.BoolVar(&dt.sift, "disable-sift", false, "Disable sift tools")
	flag.BoolVar(&dt.admin, "disable-admin", false, "Disable admin tools")
	flag.BoolVar(&dt.pyroscope, "disable-pyroscope", false, "Disable pyroscope tools")
	flag.BoolVar(&dt.navigation, "disable-navigation", false, "Disable navigation tools")
	flag.BoolVar(&dt.proxied, "disable-proxied", false, "Disable proxied tools (tools from external MCP servers)")
	flag.BoolVar(&dt.write, "disable-write", false, "Disable write tools (create/update operations)")
	flag.BoolVar(&dt.annotations, "disable-annotations", false, "Disable annotation tools")
}

func (gc *grafanaConfig) addFlags() {
	flag.BoolVar(&gc.debug, "debug", false, "Enable debug mode for the Grafana transport")

	// TLS configuration flags
	flag.StringVar(&gc.tlsCertFile, "tls-cert-file", "", "Path to TLS certificate file for client authentication")
	flag.StringVar(&gc.tlsKeyFile, "tls-key-file", "", "Path to TLS private key file for client authentication")
	flag.StringVar(&gc.tlsCAFile, "tls-ca-file", "", "Path to TLS CA certificate file for server verification")
	flag.BoolVar(&gc.tlsSkipVerify, "tls-skip-verify", false, "Skip TLS certificate verification (insecure)")
}

func (dt *disabledTools) addTools(s *server.MCPServer) {
	enabledTools := strings.Split(dt.enabledTools, ",")
	enableWriteTools := !dt.write
	maybeAddTools(s, tools.AddSearchTools, enabledTools, dt.search, "search")
	maybeAddTools(s, tools.AddDatasourceTools, enabledTools, dt.datasource, "datasource")
	maybeAddTools(s, func(mcp *server.MCPServer) { tools.AddIncidentTools(mcp, enableWriteTools) }, enabledTools, dt.incident, "incident")
	maybeAddTools(s, tools.AddPrometheusTools, enabledTools, dt.prometheus, "prometheus")
	maybeAddTools(s, tools.AddLokiTools, enabledTools, dt.loki, "loki")
	maybeAddTools(s, func(mcp *server.MCPServer) { tools.AddAlertingTools(mcp, enableWriteTools) }, enabledTools, dt.alerting, "alerting")
	maybeAddTools(s, func(mcp *server.MCPServer) { tools.AddDashboardTools(mcp, enableWriteTools) }, enabledTools, dt.dashboard, "dashboard")
	maybeAddTools(s, func(mcp *server.MCPServer) { tools.AddFolderTools(mcp, enableWriteTools) }, enabledTools, dt.folder, "folder")
	maybeAddTools(s, tools.AddOnCallTools, enabledTools, dt.oncall, "oncall")
	maybeAddTools(s, tools.AddAssertsTools, enabledTools, dt.asserts, "asserts")
	maybeAddTools(s, func(mcp *server.MCPServer) { tools.AddSiftTools(mcp, enableWriteTools) }, enabledTools, dt.sift, "sift")
	maybeAddTools(s, tools.AddAdminTools, enabledTools, dt.admin, "admin")
	maybeAddTools(s, tools.AddPyroscopeTools, enabledTools, dt.pyroscope, "pyroscope")
	maybeAddTools(s, tools.AddNavigationTools, enabledTools, dt.navigation, "navigation")
	maybeAddTools(s, func(mcp *server.MCPServer) { tools.AddAnnotationTools(mcp, enableWriteTools) }, enabledTools, dt.annotations, "annotations")
}

func newServer(transport string, dt disabledTools) (*server.MCPServer, *mcpgrafana.ToolManager) {
	sm := mcpgrafana.NewSessionManager()

	// Declare variable for ToolManager that will be initialized after server creation
	var stm *mcpgrafana.ToolManager

	// Create hooks
	hooks := &server.Hooks{
		OnRegisterSession:   []server.OnRegisterSessionHookFunc{sm.CreateSession},
		OnUnregisterSession: []server.OnUnregisterSessionHookFunc{sm.RemoveSession},
	}

	// Add proxied tools hooks if enabled and we're not running in stdio mode.
	// (stdio mode is handled by InitializeAndRegisterServerTools; per-session tools
	// are not supported).
	if transport != "stdio" && !dt.proxied {
		// OnBeforeListTools: Discover, connect, and register tools
		hooks.OnBeforeListTools = []server.OnBeforeListToolsFunc{
			func(ctx context.Context, id any, request *mcp.ListToolsRequest) {
				if stm != nil {
					if session := server.ClientSessionFromContext(ctx); session != nil {
						stm.InitializeAndRegisterProxiedTools(ctx, session)
					}
				}
			},
		}

		// OnBeforeCallTool: Fallback in case client calls tool without listing first
		hooks.OnBeforeCallTool = []server.OnBeforeCallToolFunc{
			func(ctx context.Context, id any, request *mcp.CallToolRequest) {
				if stm != nil {
					if session := server.ClientSessionFromContext(ctx); session != nil {
						stm.InitializeAndRegisterProxiedTools(ctx, session)
					}
				}
			},
		}
	}
	s := server.NewMCPServer("mcp-grafana", mcpgrafana.Version(),
		server.WithInstructions(`
This server provides access to your Grafana instance and the surrounding ecosystem.

Available Capabilities:
- Dashboards: Search, retrieve, update, and create dashboards. Extract panel queries and datasource information.
- Datasources: List and fetch details for datasources.
- Prometheus & Loki: Run PromQL and LogQL queries, retrieve metric/log metadata, and explore label names/values.
- Incidents: Search, create, update, and resolve incidents in Grafana Incident.
- Sift Investigations: Start and manage Sift investigations, analyze logs/traces, find error patterns, and detect slow requests.
- Alerting: List and fetch alert rules and notification contact points.
- OnCall: View and manage on-call schedules, shifts, teams, and users.
- Admin: List teams and perform administrative tasks.
- Pyroscope: Profile applications and fetch profiling data.
- Navigation: Generate deeplink URLs for Grafana resources like dashboards, panels, and Explore queries.
- Proxied Tools: Access tools from external MCP servers (like Tempo) through dynamic discovery.

Note that some of these capabilities may be disabled. Do not try to use features that are not available via tools.
`),
		server.WithHooks(hooks),
	)

	// Initialize ToolManager now that server is created
	stm = mcpgrafana.NewToolManager(sm, s, mcpgrafana.WithProxiedTools(!dt.proxied))

	dt.addTools(s)
	return s, stm
}

type tlsConfig struct {
	certFile, keyFile string
}

func (tc *tlsConfig) addFlags() {
	flag.StringVar(&tc.certFile, "server.tls-cert-file", "", "Path to TLS certificate file for server HTTPS (required for TLS)")
	flag.StringVar(&tc.keyFile, "server.tls-key-file", "", "Path to TLS private key file for server HTTPS (required for TLS)")
}

// httpServer represents a server with Start and Shutdown methods
type httpServer interface {
	Start(addr string) error
	Shutdown(ctx context.Context) error
}

// runHTTPServer handles the common logic for running HTTP-based servers
func runHTTPServer(ctx context.Context, srv httpServer, addr, transportName string) error {
	// Start server in a goroutine
	serverErr := make(chan error, 1)
	go func() {
		if err := srv.Start(addr); err != nil {
			serverErr <- err
		}
		close(serverErr)
	}()

	// Wait for either server error or shutdown signal
	select {
	case err := <-serverErr:
		return err
	case <-ctx.Done():
		slog.Info(fmt.Sprintf("%s server shutting down...", transportName))

		// Create a timeout context for shutdown
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()

		if err := srv.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("shutdown error: %v", err)
		}
		slog.Debug("Shutdown called, waiting for connections to close...")

		// Wait for server to finish
		select {
		case err := <-serverErr:
			// http.ErrServerClosed is expected when shutting down
			if err != nil && !errors.Is(err, http.ErrServerClosed) {
				return fmt.Errorf("server error during shutdown: %v", err)
			}
		case <-shutdownCtx.Done():
			slog.Warn(fmt.Sprintf("%s server did not stop gracefully within timeout", transportName))
		}
	}

	return nil
}

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func run(transport, addr, basePath, endpointPath string, logLevel slog.Level, dt disabledTools, gc mcpgrafana.GrafanaConfig, tls tlsConfig) error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel})))
	s, tm := newServer(transport, dt)

	// Create a context that will be cancelled on shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	// Handle shutdown signals
	go func() {
		<-sigChan
		slog.Info("Received shutdown signal")
		cancel()

		// For stdio, close stdin to unblock the Listen call
		if transport == "stdio" {
			_ = os.Stdin.Close()
		}
	}()

	// Start the appropriate server based on transport
	switch transport {
	case "stdio":
		srv := server.NewStdioServer(s)
		cf := mcpgrafana.ComposedStdioContextFunc(gc)
		srv.SetContextFunc(cf)

		// For stdio (single-tenant), initialize proxied tools on the server directly
		if !dt.proxied {
			stdioCtx := cf(ctx)
			if err := tm.InitializeAndRegisterServerTools(stdioCtx); err != nil {
				slog.Error("failed to initialize proxied tools for stdio", "error", err)
			}
		}

		slog.Info("Starting Grafana MCP server using stdio transport", "version", mcpgrafana.Version())

		err := srv.Listen(ctx, os.Stdin, os.Stdout)
		if err != nil && err != context.Canceled {
			return fmt.Errorf("server error: %v", err)
		}
		return nil

	case "sse":
		httpSrv := &http.Server{Addr: addr}
		srv := server.NewSSEServer(s,
			server.WithSSEContextFunc(mcpgrafana.ComposedSSEContextFunc(gc)),
			server.WithStaticBasePath(basePath),
			server.WithHTTPServer(httpSrv),
		)
		mux := http.NewServeMux()
		if basePath == "" {
			basePath = "/"
		}
		mux.Handle(basePath, srv)
		mux.HandleFunc("/healthz", handleHealthz)
		httpSrv.Handler = mux
		slog.Info("Starting Grafana MCP server using SSE transport",
			"version", mcpgrafana.Version(), "address", addr, "basePath", basePath)
		return runHTTPServer(ctx, srv, addr, "SSE")
	case "streamable-http":
		httpSrv := &http.Server{Addr: addr}
		opts := []server.StreamableHTTPOption{
			server.WithHTTPContextFunc(mcpgrafana.ComposedHTTPContextFunc(gc)),
			server.WithStateLess(dt.proxied), // Stateful when proxied tools enabled (requires sessions)
			server.WithEndpointPath(endpointPath),
			server.WithStreamableHTTPServer(httpSrv),
		}
		if tls.certFile != "" || tls.keyFile != "" {
			opts = append(opts, server.WithTLSCert(tls.certFile, tls.keyFile))
		}
		srv := server.NewStreamableHTTPServer(s, opts...)
		mux := http.NewServeMux()
		mux.Handle(endpointPath, srv)
		mux.HandleFunc("/healthz", handleHealthz)
		httpSrv.Handler = mux
		slog.Info("Starting Grafana MCP server using StreamableHTTP transport",
			"version", mcpgrafana.Version(), "address", addr, "endpointPath", endpointPath)
		return runHTTPServer(ctx, srv, addr, "StreamableHTTP")
	default:
		return fmt.Errorf("invalid transport type: %s. Must be 'stdio', 'sse' or 'streamable-http'", transport)
	}
}

func main() {
	var transport string
	flag.StringVar(&transport, "t", "stdio", "Transport type (stdio, sse or streamable-http)")
	flag.StringVar(
		&transport,
		"transport",
		"stdio",
		"Transport type (stdio, sse or streamable-http)",
	)
	addr := flag.String("address", "localhost:8000", "The host and port to start the sse server on")
	basePath := flag.String("base-path", "", "Base path for the sse server")
	endpointPath := flag.String("endpoint-path", "/mcp", "Endpoint path for the streamable-http server")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	showVersion := flag.Bool("version", false, "Print the version and exit")
	var dt disabledTools
	dt.addFlags()
	var gc grafanaConfig
	gc.addFlags()
	var tls tlsConfig
	tls.addFlags()
	flag.Parse()

	if *showVersion {
		fmt.Println(mcpgrafana.Version())
		os.Exit(0)
	}

	// Convert local grafanaConfig to mcpgrafana.GrafanaConfig
	grafanaConfig := mcpgrafana.GrafanaConfig{Debug: gc.debug}
	if gc.tlsCertFile != "" || gc.tlsKeyFile != "" || gc.tlsCAFile != "" || gc.tlsSkipVerify {
		grafanaConfig.TLSConfig = &mcpgrafana.TLSConfig{
			CertFile:   gc.tlsCertFile,
			KeyFile:    gc.tlsKeyFile,
			CAFile:     gc.tlsCAFile,
			SkipVerify: gc.tlsSkipVerify,
		}
	}

	if err := run(transport, *addr, *basePath, *endpointPath, parseLevel(*logLevel), dt, grafanaConfig, tls); err != nil {
		panic(err)
	}
}

func parseLevel(level string) slog.Level {
	var l slog.Level
	if err := l.UnmarshalText([]byte(level)); err != nil {
		return slog.LevelInfo
	}
	return l
}
