import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// Custom plugin to rewrite absolute paths to relative paths in dev mode
function relativePathPlugin(): Plugin {
  const basePath = process.env.VITE_BASE_PATH || '';
  
  return {
    name: 'vite-plugin-relative-path',
    transformIndexHtml(html) {
      console.log('[RelativePathPlugin] Transforming HTML...');
      // Replace absolute paths with relative paths for all Vite virtual modules
      const transformed = html
        // Handle @vite/client in script src
        .replace(/src="\/(@vite\/[^"]*)"/g, (match, path) => {
          const newPath = basePath ? `${basePath}${path}` : `./${path}`;
          console.log(`[RelativePathPlugin] Rewriting: src="/${path}" -> src="${newPath}"`);
          return `src="${newPath}"`;
        })
        // Handle @react-refresh in import statements (including inline scripts)
        .replace(/from\s+"\/(@react-refresh[^"]*)"/g, (match, path) => {
          const newPath = basePath ? `${basePath}${path}` : `./${path}`;
          console.log(`[RelativePathPlugin] Rewriting: from "/${path}" -> from "${newPath}"`);
          return `from "${newPath}"`;
        })
        // Handle @id/ virtual modules
        .replace(/from\s+"\/(@id\/[^"]*)"/g, (match, path) => {
          const newPath = basePath ? `${basePath}${path}` : `./${path}`;
          console.log(`[RelativePathPlugin] Rewriting: from "/${path}" -> from "${newPath}"`);
          return `from "${newPath}"`;
        })
        // Handle @fs/ file system access (Vite dev mode only)
        .replace(/from\s+"\/(@fs\/[^"]*)"/g, (match, path) => {
          const newPath = basePath ? `${basePath}${path}` : `./${path}`;
          console.log(`[RelativePathPlugin] Rewriting: from "/${path}" -> from "${newPath}"`);
          return `from "${newPath}"`;
        })
        // Handle any other absolute paths in href attributes
        .replace(/href="\/([^"]+)"/g, (match, path) => {
          // Don't replace if already relative or external
          if (path.startsWith('.') || path.startsWith('http')) {
            return match;
          }
          const newPath = basePath ? `${basePath}${path}` : `./${path}`;
          console.log(`[RelativePathPlugin] Rewriting: href="/${path}" -> href="${newPath}"`);
          return `href="${newPath}"`;
        })
        // Handle any remaining absolute imports in inline scripts
        .replace(/import\s+{([^}]+)}\s+from\s+"\/([^"]+)"/g, (match, imports, path) => {
          // Only rewrite if it's a Vite virtual module
          if (path.startsWith('@')) {
            const newPath = basePath ? `${basePath}${path}` : `./${path}`;
            console.log(`[RelativePathPlugin] Rewriting: import from "/${path}" -> import from "${newPath}"`);
            return `import {${imports}} from "${newPath}"`;
          }
          return match;
        });
      return transformed;
    },
    configureServer(server) {
      // Add middleware to handle @fs/ requests in proxy environments
      if (basePath) {
        server.middlewares.use((req, res, next) => {
          // Rewrite @fs/ paths to include base path if missing
          if (req.url && req.url.includes('/@fs/') && !req.url.startsWith(basePath)) {
            const originalUrl = req.url;
            // Extract the @fs/ part
            const fsMatch = req.url.match(/\/@fs\/.*/);
            if (fsMatch) {
              req.url = basePath + fsMatch[0];
              console.log(`[RelativePathPlugin] Middleware rewriting: ${originalUrl} -> ${req.url}`);
            }
          }
          next();
        });
      }
    },
  };
}

// Custom plugin to log all incoming requests
function requestLoggerPlugin(): Plugin {
  return {
    name: 'vite-plugin-request-logger',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const timestamp = new Date().toISOString();
        console.log(`[${timestamp}] ${req.method} ${req.url}`);
        next();
      });
    },
  };
}

// Get base path from environment variable
const basePath = process.env.VITE_BASE_PATH || '';

export default defineConfig({
  // Use base path from environment or empty string for relative paths
  // For Kubeflow/JupyterHub: set VITE_BASE_PATH=/notebook/user/lab/proxy/4000/
  // For local development: leave empty or set to '/'
  base: basePath,
  plugins: [
    requestLoggerPlugin(),
    react(),
    relativePathPlugin(),
  ],
  server: {
    port: 4000,
    strictPort: true,
    // HMR configuration for proxy environments
    hmr: {
      // Use the current page's protocol and host
      // This allows HMR to work through proxies
      clientPort: basePath ? undefined : 4000,
      // Note: In proxy environments, WebSocket may need special handling
      // Consider using --host 0.0.0.0 when starting the dev server
    },
    // Allow connections from any host (important for proxy environments)
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:4001',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
