import { useMemo, useState } from "react";
import { Activity } from "lucide-react";
import { createDashboard, patchDashboard } from "../api/agentApi";
import { ChatPanel, type ChatMessage } from "../components/ChatPanel";
import { DashboardFrame } from "../components/DashboardFrame";
import { demoDashboard } from "../demoDashboard";
import type { DashboardSpec } from "../types/dashboard";

const initialMessage: ChatMessage = {
  id: "assistant-welcome",
  role: "assistant",
  content: "The demo dashboard is already running with mock Prometheus data. You can ask me to modify the frame."
};

export function DashboardBuilder() {
  const [input, setInput] = useState(
    "幫我建立一個病患監控 Dashboard，使用 Line Chart 顯示 Heart Rate 和 Systolic Blood Pressure，再用 Stat Card 顯示 SpO2 與 Diastolic Blood Pressure。"
  );
  const [messages, setMessages] = useState<ChatMessage[]>([initialMessage]);
  const [dashboard, setDashboard] = useState<DashboardSpec | null>(demoDashboard);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const variables = useMemo(() => {
    if (!dashboard) return {};
    return Object.fromEntries(dashboard.variables.map((variable) => [variable.name, variable.default ?? ""]));
  }, [dashboard]);

  function addMessage(role: ChatMessage["role"], content: string) {
    setMessages((current) => [
      ...current,
      {
        id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        role,
        content
      }
    ]);
  }

  function describeDashboard(nextDashboard: DashboardSpec) {
    const widgetSummary = nextDashboard.widgets
      .map((widget) => `${widget.title} (${widget.type.replace("_", " ")})`)
      .join(", ");
    return `I updated the frame with ${nextDashboard.widgets.length} widget(s): ${widgetSummary}.`;
  }

  function describePatch(operationCount: number, nextDashboard: DashboardSpec) {
    return `I applied ${operationCount} change(s). The frame is now "${nextDashboard.title}" with ${nextDashboard.widgets.length} widget(s), version ${nextDashboard.version}.`;
  }

  async function handleSend() {
    const prompt = input.trim();
    if (!prompt || loading) return;

    addMessage("user", prompt);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      if (!dashboard) {
        const result = await createDashboard(prompt);
        setDashboard(result.dashboard);
        addMessage("assistant", describeDashboard(result.dashboard));
      } else {
        const result = await patchDashboard(dashboard.id, prompt, dashboard);
        setDashboard(result.updated_dashboard);
        addMessage("assistant", describePatch(result.patch.operations.length, result.updated_dashboard));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "The agent request failed.";
      setError(message);
      addMessage("assistant", message);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setDashboard(demoDashboard);
    setError(null);
    setMessages([initialMessage]);
  }

  return (
    <main className="appShell">
      <header className="appHeader">
        <div className="brand">
          <Activity size={26} />
          <span>Prometheus Dashboard Agent</span>
        </div>
      </header>

      <div className="workspace">
        <ChatPanel
          input={input}
          messages={messages}
          loading={loading}
          error={error}
          hasDashboard={Boolean(dashboard)}
          onInputChange={setInput}
          onSend={handleSend}
          onReset={handleReset}
        />
        <DashboardFrame dashboard={dashboard} variables={variables} />
      </div>
    </main>
  );
}
