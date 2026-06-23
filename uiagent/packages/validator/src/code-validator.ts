export type WidgetCodeValidationResult = {
  valid: boolean;
  errors: string[];
};

export class WidgetCodeValidator {
  private static FORBIDDEN_PATTERNS: Array<{ pattern: RegExp; reason: string }> = [
    { pattern: /\bwindow\b/, reason: '禁止使用 window' },
    { pattern: /\bdocument\b/, reason: '禁止使用 document' },
    { pattern: /\beval\s*\(/, reason: '禁止使用 eval()' },
    { pattern: /\bFunction\s*\(/, reason: '禁止使用 Function()' },
    { pattern: /\bXMLHttpRequest\b/, reason: '禁止使用 XMLHttpRequest' },
    { pattern: /\blocalStorage\b/, reason: '禁止使用 localStorage' },
    { pattern: /\bsessionStorage\b/, reason: '禁止使用 sessionStorage' },
    { pattern: /\bindexedDB\b/, reason: '禁止使用 indexedDB' },
    { pattern: /\bprocess\b/, reason: '禁止使用 process' },
    { pattern: /\brequire\s*\(/, reason: '禁止使用 require()' },
    { pattern: /import\s*\(/, reason: '禁止使用動態 import()' },
    { pattern: /dangerouslySetInnerHTML/, reason: '禁止使用 dangerouslySetInnerHTML' },
    { pattern: /<\s*script/i, reason: '禁止使用 <script> 標籤' },
    {
      pattern: /\bfetch\s*\(/,
      reason: '禁止直接使用 fetch()，請改用 props.fetchData() 注入資料來源',
    },
  ];

  private static ALLOWED_IMPORT_PREFIXES = ['react', '@ui-agent/types', 'recharts'];

  validate(code: string): WidgetCodeValidationResult {
    const errors: string[] = [];

    if (!code || !code.trim()) {
      return {
        valid: false,
        errors: ['程式碼不可為空'],
      };
    }

    for (const { pattern, reason } of WidgetCodeValidator.FORBIDDEN_PATTERNS) {
      if (pattern.test(code)) errors.push(reason);
    }

    const importRegex = /from\s+['"]([^'"]+)['"]/g;
    const imports = Array.from(code.matchAll(importRegex)).map((m) => m[1]);
    for (const source of imports) {
      const allowed = WidgetCodeValidator.ALLOWED_IMPORT_PREFIXES.some((p) =>
        source.startsWith(p)
      );
      if (!allowed) {
        errors.push(`不允許的 import 來源: ${source}`);
      }
    }

    if (!/export\s+default\s+/m.test(code)) {
      errors.push('Widget 程式碼必須包含 default export');
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}

