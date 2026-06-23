/**
 * @ui-agent/validator
 * UI Spec validation and sanitization
 */

export { UISpecValidator, validateUISpec } from './validator';
export { uiSpecSchema } from './schema';
export { WidgetCodeValidator } from './code-validator';
export {
  canRenderUISpec,
  BUILTIN_CAPABILITY_CATALOG,
  strategyFromCoverage,
} from './coverage-checker';
export type { WidgetCodeValidationResult } from './code-validator';
export type {
  CapabilityCatalog,
  CoverageDiagnostics,
  WidgetCapability,
  RiskLevel,
} from './coverage-checker';
