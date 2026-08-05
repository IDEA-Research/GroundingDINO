/**
 * UI Spec Validator
 * Validates UI specifications against schema and business rules
 */

import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import type {
  UISpec,
  ValidationResult,
  ValidationError,
  ValidationErrorType,
} from '@ui-agent/types';
import { uiSpecSchema } from './schema';

export class UISpecValidator {
  private ajv: Ajv;
  private validateSpec: any;

  constructor() {
    this.ajv = new Ajv({
      allErrors: true,
      verbose: true,
      strict: false,
    });
    addFormats(this.ajv);
    this.validateSpec = this.ajv.compile(uiSpecSchema);
  }

  /**
   * Validate a UI Spec against schema and business rules
   */
  validate(spec: unknown): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: string[] = [];

    // Step 1: Schema validation
    const schemaValid = this.validateSpec(spec);
    if (!schemaValid && this.validateSpec.errors) {
      for (const error of this.validateSpec.errors) {
        errors.push({
          type: 'SCHEMA_VIOLATION',
          message: `${error.instancePath} ${error.message}`,
          path: error.instancePath,
        });
      }
    }

    if (errors.length > 0) {
      return { valid: false, errors, warnings };
    }

    const typedSpec = spec as UISpec;

    // Step 2: Business rule validation
    this.validateWidgetPositions(typedSpec, errors);
    this.validateTimeRanges(typedSpec, errors, warnings);
    this.validateQueries(typedSpec, errors);
    this.validateWidgetIds(typedSpec, errors);

    return {
      valid: errors.length === 0,
      errors,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  }

  /**
   * Validate widget positions don't overlap (for grid layout)
   */
  private validateWidgetPositions(spec: UISpec, errors: ValidationError[]): void {
    if (spec.layout.type !== 'grid') return;

    const positions = new Map<string, string>();

    for (const widget of spec.widgets) {
      const { row, col, colspan = 1, rowspan = 1 } = widget.position;

      // Check all cells this widget occupies
      for (let r = row; r < row + rowspan; r++) {
        for (let c = col; c < col + colspan; c++) {
          const key = `${r},${c}`;
          const existingWidget = positions.get(key);

          if (existingWidget) {
            errors.push({
              type: 'POSITION_CONFLICT',
              message: `Widget ${widget.id} overlaps with ${existingWidget} at position (${r},${c})`,
              widget_id: widget.id,
            });
          } else {
            positions.set(key, widget.id);
          }
        }
      }
    }
  }

  /**
   * Validate time ranges are reasonable
   */
  private validateTimeRanges(
    spec: UISpec,
    errors: ValidationError[],
    warnings: string[]
  ): void {
    const MAX_RANGE_DAYS = 90;

    for (const widget of spec.widgets) {
      const { start, end } = widget.data_source.time_range;

      try {
        const startTime = this.parseTime(start);
        const endTime = this.parseTime(end);

        if (startTime >= endTime) {
          errors.push({
            type: 'INVALID_TIME_RANGE',
            message: `Start time must be before end time`,
            widget_id: widget.id,
            path: `widgets[${widget.id}].time_range`,
          });
        }

        const rangeDays = (endTime - startTime) / (1000 * 60 * 60 * 24);
        if (rangeDays > MAX_RANGE_DAYS) {
          warnings.push(
            `Widget ${widget.id}: Time range of ${rangeDays.toFixed(1)} days exceeds recommended ${MAX_RANGE_DAYS} days. Query may be slow.`
          );
        }
      } catch (error) {
        errors.push({
          type: 'INVALID_TIME_RANGE',
          message: `Invalid time format: ${error instanceof Error ? error.message : String(error)}`,
          widget_id: widget.id,
        });
      }
    }
  }

  /**
   * Parse time string to timestamp
   */
  private parseTime(timeStr: string): number {
    // Handle relative time (e.g., "now-1h")
    if (timeStr.startsWith('now')) {
      const now = Date.now();
      if (timeStr === 'now') return now;

      const match = timeStr.match(/^now-(\d+)([smhd])$/);
      if (!match) {
        throw new Error(`Invalid relative time format: ${timeStr}`);
      }

      const value = parseInt(match[1], 10);
      const unit = match[2];

      const multipliers: Record<string, number> = {
        s: 1000,
        m: 60 * 1000,
        h: 60 * 60 * 1000,
        d: 24 * 60 * 60 * 1000,
      };

      return now - value * multipliers[unit];
    }

    // Handle ISO 8601
    const timestamp = Date.parse(timeStr);
    if (isNaN(timestamp)) {
      throw new Error(`Invalid ISO 8601 time: ${timeStr}`);
    }
    return timestamp;
  }

  /**
   * Validate PromQL queries for safety
   */
  private validateQueries(spec: UISpec, errors: ValidationError[]): void {
    const FORBIDDEN_PATTERNS = [
      /delete/i,
      /drop/i,
      /—exec/i,
      /\beval\b/i,
    ];

    const MAX_QUERY_LENGTH = 10000;

    for (const widget of spec.widgets) {
      const query = widget.data_source.query;

      // Check length
      if (query.length > MAX_QUERY_LENGTH) {
        errors.push({
          type: 'INVALID_QUERY',
          message: `Query exceeds maximum length of ${MAX_QUERY_LENGTH} characters`,
          widget_id: widget.id,
        });
      }

      // Check for forbidden patterns
      for (const pattern of FORBIDDEN_PATTERNS) {
        if (pattern.test(query)) {
          errors.push({
            type: 'INVALID_QUERY',
            message: `Query contains forbidden pattern: ${pattern.source}`,
            widget_id: widget.id,
          });
        }
      }

      // Basic PromQL syntax check (simple validation)
      if (!this.isValidPromQLSyntax(query)) {
        errors.push({
          type: 'INVALID_QUERY',
          message: `Query appears to have invalid PromQL syntax`,
          widget_id: widget.id,
        });
      }
    }
  }

  /**
   * Basic PromQL syntax validation
   */
  private isValidPromQLSyntax(query: string): boolean {
    // Check for balanced parentheses and brackets
    const parenBalance = this.checkBalance(query, '(', ')');
    const bracketBalance = this.checkBalance(query, '[', ']');
    const braceBalance = this.checkBalance(query, '{', '}');

    return parenBalance && bracketBalance && braceBalance;
  }

  /**
   * Check if brackets/parentheses are balanced
   */
  private checkBalance(str: string, open: string, close: string): boolean {
    let count = 0;
    for (const char of str) {
      if (char === open) count++;
      if (char === close) count--;
      if (count < 0) return false;
    }
    return count === 0;
  }

  /**
   * Validate widget IDs are unique
   */
  private validateWidgetIds(spec: UISpec, errors: ValidationError[]): void {
    const ids = new Set<string>();

    for (const widget of spec.widgets) {
      if (ids.has(widget.id)) {
        errors.push({
          type: 'SCHEMA_VIOLATION',
          message: `Duplicate widget ID: ${widget.id}`,
          widget_id: widget.id,
        });
      }
      ids.add(widget.id);
    }
  }
}

/**
 * Convenience function to validate a UI Spec
 */
export function validateUISpec(spec: unknown): ValidationResult {
  const validator = new UISpecValidator();
  return validator.validate(spec);
}
