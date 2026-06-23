import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PromptInput from '../components/PromptInput';
import type { ValidationError } from '@ui-agent/types';
import { generateDashboard } from '../services/dashboard-api';

function GeneratePage() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationError[] | null>(null);
  const [lastPrompt, setLastPrompt] = useState<string>('');

  const handlePromptSubmit = async (prompt: string) => {
    setIsLoading(true);
    setError(null);
    setValidationErrors(null);
    setLastPrompt(prompt);

    try {
      const data = await generateDashboard(prompt);
      navigate(
        `/dashboards/${encodeURIComponent(data.dashboardId)}?version=${encodeURIComponent(data.versionToken)}`
      );
    } catch (err: unknown) {
      const data = err as { validationErrors?: ValidationError[]; message?: string };
      if (data.validationErrors) {
        setValidationErrors(data.validationErrors);
        setError(data.message || 'UI Spec validation 失敗');
      } else {
        setValidationErrors(null);
        setError(data.message || 'Failed to generate UI spec');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">UI Agent</h1>
          <p className="text-lg text-gray-600">使用自然語言生成 Prometheus 監控儀表板</p>
        </header>

        <div className="max-w-6xl mx-auto">
          <PromptInput
            onSubmit={handlePromptSubmit}
            isLoading={isLoading}
            error={error}
            validationErrors={validationErrors}
            initialPrompt={lastPrompt}
          />
        </div>
      </div>
    </div>
  );
}

export default GeneratePage;

