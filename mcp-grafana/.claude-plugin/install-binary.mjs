#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync, chmodSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import process from 'node:process';
import { pipeline } from 'node:stream/promises';

const PLUGIN_ROOT = process.env.CLAUDE_PLUGIN_ROOT;
if (!PLUGIN_ROOT) {
  console.error('Error: CLAUDE_PLUGIN_ROOT environment variable not set');
  process.exit(1);
}

// Detect OS and architecture
const platform = process.platform;
const arch = process.arch;

let OS, ARCH, EXT, BINARY_NAME, BINARY_PATH;

// Map Node.js arch to release naming
switch (arch) {
  case 'x64':
    ARCH = 'x86_64';
    break;
  case 'arm64':
    ARCH = 'arm64';
    break;
  case 'ia32':
    ARCH = 'i386';
    break;
  default:
    console.error(`Unsupported architecture: ${arch}`);
    process.exit(1);
}

// Map Node.js platform to release naming
switch (platform) {
  case 'darwin':
    OS = 'Darwin';
    EXT = 'tar.gz';
    BINARY_NAME = 'mcp-grafana';
    break;
  case 'linux':
    OS = 'Linux';
    EXT = 'tar.gz';
    BINARY_NAME = 'mcp-grafana';
    break;
  case 'win32':
    OS = 'Windows';
    EXT = 'zip';
    BINARY_NAME = 'mcp-grafana.exe';
    break;
  default:
    console.error(`Unsupported OS: ${platform}`);
    process.exit(1);
}

BINARY_PATH = join(PLUGIN_ROOT, BINARY_NAME);

// Fetch latest version from GitHub API
async function getLatestVersion() {
  const headers = {};
  // Use GitHub token if available (for CI environments)
  if (process.env.GITHUB_TOKEN) {
    headers['Authorization'] = `Bearer ${process.env.GITHUB_TOKEN}`;
  }

  const response = await fetch('https://api.github.com/repos/grafana/mcp-grafana/releases/latest', {
    headers
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch latest version: ${response.statusText}`);
  }
  const data = await response.json();
  return data.tag_name;
}

// Download file from URL
async function downloadFile(url, destPath) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.statusText}`);
  }
  await pipeline(response.body, createWriteStream(destPath));
}

// Verify SHA256 checksum
async function verifyChecksum(filePath, checksumsContent, archiveName) {
  const fileBuffer = readFileSync(filePath);
  const hash = createHash('sha256').update(fileBuffer).digest('hex');

  const lines = checksumsContent.split('\n');
  for (const line of lines) {
    if (line.includes(archiveName)) {
      const [expectedHash] = line.split(/\s+/);
      if (hash === expectedHash) {
        console.error(`✓ Checksum verified`);
        return true;
      } else {
        throw new Error(`Checksum mismatch for ${archiveName}`);
      }
    }
  }
  throw new Error(`No checksum found for ${archiveName}`);
}

// Extract tar.gz archive using system tar command
function extractTarGz(archivePath, destDir) {
  return new Promise((resolve, reject) => {
    const tar = spawn('tar', ['-xzf', archivePath, '-C', destDir]);
    tar.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`tar extraction failed with code ${code}`));
    });
  });
}

// Extract zip archive using system command
function extractZip(archivePath, destDir) {
  return new Promise((resolve, reject) => {
    // Use PowerShell on Windows
    const unzip = spawn('powershell', ['-Command', `Expand-Archive -Path "${archivePath}" -DestinationPath "${destDir}" -Force`]);
    unzip.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`zip extraction failed with code ${code}`));
    });
  });
}

async function main() {
  try {
    // Get latest version
    console.error('Fetching latest version...');
    const VERSION = await getLatestVersion();

    const ARCHIVE_NAME = `mcp-grafana_${OS}_${ARCH}.${EXT}`;
    const VERSION_FILE = join(PLUGIN_ROOT, '.mcp-grafana-version');

    // Check if binary exists and version matches
    const needsInstall = !existsSync(BINARY_PATH) ||
                         !existsSync(VERSION_FILE) ||
                         readFileSync(VERSION_FILE, 'utf8').trim() !== VERSION;

    if (!needsInstall) {
      // Binary is up to date, just execute it
      const child = spawn(BINARY_PATH, process.argv.slice(2), { stdio: 'inherit' });
      child.on('exit', (code) => process.exit(code || 0));
      return;
    }

    console.error(`Downloading mcp-grafana ${VERSION} for ${OS}-${ARCH}...`);

    // Create temp directory
    const TEMP_DIR = join(tmpdir(), `mcp-grafana-${Date.now()}`);
    mkdirSync(TEMP_DIR, { recursive: true });

    try {
      const ARCHIVE_PATH = join(TEMP_DIR, ARCHIVE_NAME);
      const DOWNLOAD_URL = `https://github.com/grafana/mcp-grafana/releases/latest/download/${ARCHIVE_NAME}`;

      // Download archive
      await downloadFile(DOWNLOAD_URL, ARCHIVE_PATH);

      // Download and verify checksums
      console.error('Verifying checksum...');
      const VERSION_NUMBER = VERSION.replace(/^v/, ''); // Remove 'v' prefix
      const CHECKSUMS_URL = `https://github.com/grafana/mcp-grafana/releases/download/${VERSION}/mcp-grafana_${VERSION_NUMBER}_checksums.txt`;
      const checksumResponse = await fetch(CHECKSUMS_URL);
      if (!checksumResponse.ok) {
        throw new Error(`Failed to download checksums: ${checksumResponse.statusText}`);
      }
      const checksumsContent = await checksumResponse.text();
      await verifyChecksum(ARCHIVE_PATH, checksumsContent, ARCHIVE_NAME);

      // Extract archive
      console.error('Extracting archive...');
      if (EXT === 'tar.gz') {
        await extractTarGz(ARCHIVE_PATH, TEMP_DIR);
      } else {
        await extractZip(ARCHIVE_PATH, TEMP_DIR);
      }

      // Move binary to plugin root
      const extractedBinary = join(TEMP_DIR, BINARY_NAME);
      if (!existsSync(extractedBinary)) {
        throw new Error(`Binary not found after extraction: ${extractedBinary}`);
      }

      mkdirSync(PLUGIN_ROOT, { recursive: true });
      const binaryContent = readFileSync(extractedBinary);
      writeFileSync(BINARY_PATH, binaryContent);

      if (platform !== 'win32') {
        chmodSync(BINARY_PATH, 0o755);
      }

      writeFileSync(VERSION_FILE, VERSION);

      console.error(`Successfully installed mcp-grafana ${VERSION}`);
    } finally {
      // Cleanup temp directory
      try {
        const { rmSync } = await import('fs');
        rmSync(TEMP_DIR, { recursive: true, force: true });
      } catch (e) {
        // Ignore cleanup errors
      }
    }

    // Execute the binary
    const child = spawn(BINARY_PATH, process.argv.slice(2), { stdio: 'inherit' });
    child.on('exit', (code) => process.exit(code || 0));

  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

main();