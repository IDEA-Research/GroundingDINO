/**
 * AI Agent Types
 */

import type { UISpec } from './ui-spec';

// ============================================
// Agent Request/Response Types
// ============================================

export type AgentIntent = {
  action: 'generate_ui' | 'modify_ui' | 'explain_ui' | 'suggest';
  query: string;
  context?: AgentContext;
};

export type AgentContext = {
  user_id?: string;
  session_id?: string;
  timezone?: string;
  locale?: string;
  previous_specs?: UISpec[];
  user_preferences?: UserPreferences;
};

export type UserPreferences = {
  default_time_range?: string;
  preferred_chart_type?: string;
  color_scheme?: string;
  auto_refresh_enabled?: boolean;
};

export type AgentResponse = {
  success: boolean;
  spec?: UISpec;
  widget_patches?: WidgetCodePatch[];
  message?: string;
  suggestions?: string[];
  confidence?: number; // 0-1
  reasoning?: string; // Explanation of the AI's decision
  strategy_used?: GenerationStrategyMode;
};

export type GenerationStrategyMode =
  | 'spec_only'
  | 'inherit_widget'
  | 'new_widget';

export type GenerationStrategy =
  | { mode: 'spec_only' }
  | { mode: 'inherit_widget'; base_widget: string }
  | { mode: 'new_widget' };

export type WidgetCodePatch = {
  widget_type: string;
  base_widget?: string;
  code: string;
  config_schema?: Record<string, unknown>;
  description: string;
  generated_file_path?: string;
  compile_success?: boolean;
  compile_output?: string;
  compile_attempts?: number;
};

export type PlannerResult = {
  strategy: GenerationStrategy;
  reasoning: string;
  widget_analysis: {
    required_capabilities: string[];
    existing_coverage: string[];
    gaps: string[];
  };
};

// ============================================
// Prompt Templates
// ============================================

export type PromptTemplate = {
  id: string;
  name: string;
  system_prompt: string;
  few_shot_examples: FewShotExample[];
  constraints: string[];
};

export type FewShotExample = {
  user_query: string;
  expected_output: UISpec;
  reasoning?: string;
};

// ============================================
// Intent Classification
// ============================================

export type ClassifiedIntent = {
  primary_intent:
    | 'view_metrics'
    | 'monitor_alerts'
    | 'compare_resources'
    | 'analyze_trend'
    | 'troubleshoot';
  data_requirements: DataRequirement[];
  suggested_widgets: string[];
  time_context: TimeContext;
};

export type DataRequirement = {
  metric_name: string;
  labels?: Record<string, string>;
  aggregation?: 'avg' | 'sum' | 'max' | 'min' | 'rate' | 'count';
  grouping?: string[];
};

export type TimeContext = {
  relative_range?: string; // "now-1h", "now-24h"
  absolute_range?: {
    start: string;
    end: string;
  };
  granularity?: string; // "1m", "5m", "1h"
};

// ============================================
// Agent State & Memory
// ============================================

export type ConversationTurn = {
  timestamp: string;
  user_message: string;
  agent_response: AgentResponse;
  spec_generated?: UISpec;
};

export type AgentMemory = {
  session_id: string;
  user_id: string;
  conversation_history: ConversationTurn[];
  generated_specs: UISpec[];
  user_feedback: UserFeedback[];
};

export type UserFeedback = {
  spec_id: string;
  rating: 1 | 2 | 3 | 4 | 5;
  comment?: string;
  timestamp: string;
};
