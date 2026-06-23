/**
 * @ui-agent/agent
 * AI Agent for generating UI specifications
 */

export {
  SYSTEM_PROMPT,
  FEW_SHOT_EXAMPLES,
  PROMETHEUS_MONITORING_TEMPLATE,
  buildPrompt,
} from './prompts';

export { planGenerationStrategy } from './planner';

export {
  PLANNER_SYSTEM_PROMPT,
  WIDGET_CODE_SYSTEM_PROMPT,
  buildWidgetCodePrompt,
} from './widget-code-prompts';
