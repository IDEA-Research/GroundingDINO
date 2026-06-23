import type { KeyboardEvent } from "react";
import { Loader2, RotateCcw, Send, Sparkles } from "lucide-react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface Props {
  input: string;
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  hasDashboard: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onReset: () => void;
}

const promptChips = [
  "Show Heart Rate and SpO2 as live trends",
  "Add systolic and diastolic blood pressure cards",
  "Change refresh to 10 seconds",
  "Add alerts for abnormal vitals"
];

export function ChatPanel({
  input,
  messages,
  loading,
  error,
  hasDashboard,
  onInputChange,
  onSend,
  onReset
}: Props) {
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  }

  return (
    <aside className="chatPanel">
      <div className="chatHeader">
        <div className="panelTitle">
          <Sparkles size={18} />
          <span>LLM Console</span>
        </div>
        <button className="iconButton" onClick={onReset} disabled={loading || !hasDashboard} title="Reset">
          <RotateCcw size={18} />
        </button>
      </div>

      <div className="messageList">
        {messages.map((message) => (
          <div key={message.id} className={`messageBubble ${message.role}`}>
            <span>{message.role === "user" ? "You" : "Agent"}</span>
            <p>{message.content}</p>
          </div>
        ))}
        {loading && (
          <div className="messageBubble assistant">
            <span>Agent</span>
            <p className="loadingLine">
              <Loader2 size={16} />
              Thinking
            </p>
          </div>
        )}
      </div>

      <div className="chipRow">
        {promptChips.map((chip) => (
          <button key={chip} className="chipButton" onClick={() => onInputChange(chip)} disabled={loading}>
            {chip}
          </button>
        ))}
      </div>

      {error && <div className="chatError">{error}</div>}

      <div className="composer">
        <textarea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask for the vitals, charts, cards, thresholds, time range, or refresh interval."
        />
        <button onClick={onSend} disabled={loading || !input.trim()}>
          <Send size={18} />
          Send
        </button>
      </div>
    </aside>
  );
}
