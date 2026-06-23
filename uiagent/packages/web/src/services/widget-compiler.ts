import React from 'react';
import type { WidgetCodePatch } from '@ui-agent/types';

type RuntimeWidgetModule = {
  default?: React.ComponentType<{ widget: any }>;
};

const CODE_FENCE_REGEX = /```(?:javascript|js|tsx|ts|jsx)?\s*([\s\S]*?)```/i;

function normalizePatchCode(rawCode: string): string {
  // 優先擷取 markdown code fence 內文，避免前後說明文字混入
  const fenced = rawCode.match(CODE_FENCE_REGEX)?.[1] ?? rawCode;

  return fenced
    .replace(/^\uFEFF/, '') // BOM
    .replace(/[\u200B-\u200D\u2060]/g, '') // zero-width chars
    .trim();
}

function detectUnsupportedSyntax(code: string): string | null {
  // new Function 無法直接編譯 ESM import/export 與原生 JSX
  if (/^\s*import\s/m.test(code)) {
    return '包含 import 語法（runtime new Function 不支援 ESM import）';
  }

  if (/^\s*export\s/m.test(code)) {
    return '包含 export 語法（請改為 module.exports.default）';
  }

  if (/return\s*\(\s*<|=\s*<\w+/m.test(code)) {
    return '疑似包含 JSX（runtime 期待可直接執行的 JS，而非 TSX/JSX）';
  }

  return null;
}

function previewCode(code: string, maxLines = 8): string {
  return code
    .split('\n')
    .slice(0, maxLines)
    .map((line, idx) => `${idx + 1}: ${line}`)
    .join('\n');
}

/**
 * MVP runtime compiler:
 * - 以受限參數注入方式執行程式碼
 * - 期待 server 產生可直接執行 JS（非 TSX）字串
 */
export function compileWidgetCode(
  patch: WidgetCodePatch
): React.ComponentType<{ widget: any }> {
  const sanitizedCode = normalizePatchCode(patch.code);

  const unsupportedReason = detectUnsupportedSyntax(sanitizedCode);
  if (unsupportedReason) {
    throw new SyntaxError(
      `Widget ${patch.widget_type} patch 無法編譯：${unsupportedReason}\n` +
        `Code preview:\n${previewCode(sanitizedCode)}`
    );
  }

  let mod: RuntimeWidgetModule;
  try {
    const factory = new Function(
      'React',
      `
        const module = { exports: {} };
        const exports = module.exports;
        ${sanitizedCode}
        return module.exports;
      `
    ) as (react: typeof React) => RuntimeWidgetModule;

    mod = factory(React);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new SyntaxError(
      `Widget ${patch.widget_type} patch 編譯失敗：${message}\n` +
        `Code preview:\n${previewCode(sanitizedCode)}`
    );
  }
  const component = mod.default;

  if (!component) {
    throw new Error(`Widget ${patch.widget_type} 沒有 default export`);
  }

  return component;
}
