import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// 獲取當前文件所在目錄
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 載入 packages/api/.env 文件 - 必須在其他導入之前
dotenv.config({ path: join(__dirname, '../.env') });

// 導入其他模組（在環境變數載入之後）
import express, { Express } from 'express';
import cors from 'cors';
import {
  generateUISpec,
  fixUISpec,
  generateWidgetPatches,
} from './services/generator.js';
import { dashboardStore } from './services/dashboard-store.js';
import { planGenerationStrategy } from '@ui-agent/agent';
import {
  validateUISpec,
  WidgetCodeValidator,
  canRenderUISpec,
  BUILTIN_CAPABILITY_CATALOG,
  strategyFromCoverage,
} from '@ui-agent/validator';
import { PrometheusExecutor } from '@ui-agent/data-source';
import type { PrometheusDataSource, UISpec } from '@ui-agent/types';

const app: Express = express();
const PORT = process.env.PORT || 4001;

// Initialize Prometheus Executor
const prometheusUrl = process.env.PROMETHEUS_URL || 'http://localhost:9090';
const prometheusExecutor = new PrometheusExecutor({ prometheusUrl });
const widgetCodeValidator = new WidgetCodeValidator();

function normalizeUISpec(spec: UISpec): UISpec {
  const normalized = structuredClone(spec) as UISpec;

  const controlWidgetTypes = new Set([
    'time_range_buttons',
    'time_range_picker',
    'time_options',
  ]);
  const movedControls = normalized.widgets.filter((w) => controlWidgetTypes.has(w.type));

  if (movedControls.length > 0) {
    normalized.widgets = normalized.widgets.filter((w) => !controlWidgetTypes.has(w.type));

    const existingActions = normalized.actions ?? [];
    const hasTimeRangePicker = existingActions.some((a) => a.type === 'time_range_picker');

    if (!hasTimeRangePicker) {
      existingActions.push({
        id: 'action-time-range-picker',
        type: 'time_range_picker',
        label: '時間範圍',
        config: {
          presets: ['1h', '24h', '7d'],
        },
      });
    }

    normalized.actions = existingActions;
  }

  return normalized;
}

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Generate UI Spec endpoint with auto-fix retry logic
app.post('/api/generate', async (req, res) => {
  const MAX_RETRIES = 10;
  
  try {
    const { prompt } = req.body;

    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'Prompt is required and must be a string',
      });
    }

    console.log('Generating UI Spec for prompt:', prompt);

    // Plan strategy first (prompt-based heuristic plan)
    const plan = planGenerationStrategy(prompt);
    console.log('Generation strategy:', plan.strategy.mode, plan.reasoning);

    // Generate initial UI Spec using AI
    let uiSpec: UISpec = normalizeUISpec(await generateUISpec(prompt));
    let validation = validateUISpec(uiSpec);
    let retryCount = 0;

    // Auto-fix retry loop
    while (!validation.valid && retryCount < MAX_RETRIES) {
      retryCount++;
      console.log(`\n=== RETRY ${retryCount}/${MAX_RETRIES} ===`);
      console.error('UI Spec validation failed:', validation.errors);
      
      try {
        // Try to fix the UI Spec using AI
        console.log('Attempting to fix UI Spec...');
        uiSpec = normalizeUISpec(await fixUISpec(prompt, uiSpec, validation.errors));
        
        // Re-validate the fixed UI Spec
        validation = validateUISpec(uiSpec);
        
        if (validation.valid) {
          console.log(`✓ UI Spec fixed successfully on retry ${retryCount}`);
        }
      } catch (fixError) {
        console.error(`Failed to fix UI Spec on retry ${retryCount}:`, fixError);
        // If fix fails, break the loop and return the error
        break;
      }
    }

    // Check final validation status
    if (!validation.valid) {
      console.error(`Failed to generate valid UI Spec after ${retryCount} retries`);
      return res.status(500).json({
        error: 'UI Spec generation failed',
        message: `Generated UI Spec is invalid after ${retryCount} attempts`,
        validationErrors: validation.errors,
        retryCount,
      });
    }

    // Success
    if (validation.warnings && validation.warnings.length > 0) {
      console.warn('UI Spec warnings:', validation.warnings);
    }

    console.log(`✓ Successfully generated valid UI Spec${retryCount > 0 ? ` (after ${retryCount} retries)` : ''}`);

    // Coverage checker + strategy gate (Phase A + B)
    let coverageDiagnostics = canRenderUISpec(uiSpec, BUILTIN_CAPABILITY_CATALOG);

    // If actions/controls are misplaced in widgets, normalize then re-check once
    const hasMisplacedControls = coverageDiagnostics.unsupportedWidgetTypes.some((type) =>
      ['time_range_buttons', 'time_range_picker', 'time_options'].includes(type)
    );
    if (hasMisplacedControls) {
      uiSpec = normalizeUISpec(uiSpec);
      coverageDiagnostics = canRenderUISpec(uiSpec, BUILTIN_CAPABILITY_CATALOG);
    }

    const gatedStrategy = strategyFromCoverage(
      coverageDiagnostics,
      plan.strategy.mode === 'inherit_widget' ? plan.strategy.base_widget : 'metric_card'
    );
    console.log(
      'Coverage diagnostics:',
      coverageDiagnostics.coverageScore,
      coverageDiagnostics.riskLevel,
      '=> strategy gate:',
      gatedStrategy.mode
    );

    // Generate widget patches for non-spec-only strategy
    let widgetPatches = await generateWidgetPatches(prompt, gatedStrategy);
    if (widgetPatches.length > 0) {
      for (const patch of widgetPatches) {
        const validationResult = widgetCodeValidator.validate(patch.code);
        if (!validationResult.valid) {
          console.warn(
            `Invalid widget patch for ${patch.widget_type}, fallback to spec-only:`,
            validationResult.errors
          );
          widgetPatches = [];
          break;
        }
      }

      // align uiSpec widget type with generated widget type when applicable
      const generatedType = widgetPatches[0]?.widget_type;
      if (generatedType && gatedStrategy.mode !== 'spec_only' && uiSpec.widgets.length > 0) {
        uiSpec.widgets[0].type = generatedType as typeof uiSpec.widgets[0]['type'];
      }
    }
    
    const dashboardRecord = dashboardStore.save({
      uiSpec,
      widgetPatches,
    });

    res.json({
      success: true,
      dashboardId: dashboardRecord.dashboardId,
      versionToken: dashboardRecord.versionToken,
      updatedAt: dashboardRecord.updatedAt,
      uiSpec,
      warnings: validation.warnings,
      retryCount, // Include retry count in response for debugging
      strategyUsed: gatedStrategy.mode,
      reasoning: plan.reasoning,
      coverageDiagnostics,
      widgetPatches,
      widget_patches: widgetPatches,
    });
  } catch (error) {
    console.error('Error generating UI Spec:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.get('/api/dashboards/:id/meta', (req, res) => {
  const dashboardId = req.params.id;
  const record = dashboardStore.get(dashboardId);

  if (!record) {
    return res.status(404).json({
      error: 'Dashboard not found',
      message: `Dashboard ${dashboardId} does not exist`,
    });
  }

  res.json({
    success: true,
    dashboardId: record.dashboardId,
    versionToken: record.versionToken,
    updatedAt: record.updatedAt,
  });
});

app.get('/api/dashboards/:id', (req, res) => {
  const dashboardId = req.params.id;
  const record = dashboardStore.get(dashboardId);

  if (!record) {
    return res.status(404).json({
      error: 'Dashboard not found',
      message: `Dashboard ${dashboardId} does not exist`,
    });
  }

  res.json({
    success: true,
    dashboardId: record.dashboardId,
    versionToken: record.versionToken,
    updatedAt: record.updatedAt,
    uiSpec: record.uiSpec,
    widgetPatches: record.widgetPatches,
  });
});

// Prometheus Query endpoint
app.post('/api/query', async (req, res) => {
  try {
    const { dataSource } = req.body as { dataSource: PrometheusDataSource };

    if (!dataSource || dataSource.type !== 'prometheus') {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'Valid Prometheus data source is required',
      });
    }

    console.log('Executing Prometheus query:', dataSource.query);

    // Execute query using PrometheusExecutor
    const result = await prometheusExecutor.execute(dataSource);

    res.json({
      success: true,
      data: result,
    });
  } catch (error) {
    console.error('Error executing Prometheus query:', error);
    res.status(500).json({
      error: 'Query execution failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 UI Agent API server running on http://localhost:${PORT}`);
  console.log(`📝 API endpoint: http://localhost:${PORT}/api/generate`);
  
  // 顯示 API Key 狀態
  const hasApiKey = !!process.env.OPENAI_API_KEY;
  if (hasApiKey) {
    const keyPreview = process.env.OPENAI_API_KEY!.substring(0, 10) + '...';
    console.log(`🔑 OpenAI API Key: ${keyPreview} (已載入)`);
  } else {
    console.log(`⚠️  OpenAI API Key 未設定 - 將使用 mock 資料`);
  }
  
  // 顯示 Prometheus URL 狀態
  const prometheusUrl = process.env.PROMETHEUS_URL || 'http://localhost:9090';
  console.log(`📊 Prometheus Server: ${prometheusUrl}`);
});

export default app;
