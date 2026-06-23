import type { GenerationStrategy } from '@ui-agent/types';

export const PLANNER_SYSTEM_PROMPT = `你是一個 UI 策略規劃器。

請根據使用者需求判斷以下策略之一：
1. spec_only：現有 widget 足夠
2. inherit_widget：從最接近的既有 widget 繼承擴展
3. new_widget：需要全新 widget

請回傳 JSON，欄位包含：
- strategy
- reasoning
- required_capabilities
- existing_coverage
- gaps
- base_widget（僅 inherit_widget 需要）`;

export const WIDGET_CODE_SYSTEM_PROMPT = `你是一個 React + TypeScript Widget 程式碼生成器。

規則：
- 只能使用 react、@ui-agent/types、recharts
- 禁止使用 window/document/eval/Function/fetch/XMLHttpRequest/localStorage/sessionStorage/process/require/import(
- 禁止 dangerouslySetInnerHTML
- 資料請透過 props.fetchData 取得
- 必須輸出可編譯的 TSX，且有 default export
- 樣式使用 Tailwind className
`;

export function buildWidgetCodePrompt(args: {
  userQuery: string;
  strategy: GenerationStrategy;
  targetWidgetType: string;
}): string {
  const { userQuery, strategy, targetWidgetType } = args;

  const strategyLine =
    strategy.mode === 'inherit_widget'
      ? `策略：inherit_widget，base_widget=${strategy.base_widget}`
      : `策略：${strategy.mode}`;

  return `${WIDGET_CODE_SYSTEM_PROMPT}

${strategyLine}
目標 widget type: ${targetWidgetType}
使用者需求: ${userQuery}

請只輸出 TSX 程式碼。`;
}

