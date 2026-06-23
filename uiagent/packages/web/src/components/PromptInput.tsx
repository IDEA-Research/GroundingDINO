import React, { useState, useEffect } from 'react';
import type { ValidationError } from '@ui-agent/types';

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
  isLoading: boolean;
  error: string | null;
  validationErrors?: ValidationError[] | null;
  initialPrompt?: string;
}

const EXAMPLE_PROMPTS = [
  '顯示過去 1 小時的 CPU 使用率',
  '我想看現在記憶體使用了多少',
  '顯示網路流量最高的 5 台機器',
  '過去 24 小時磁碟使用率的變化',
  '顯示過去 1 小時的 CPU 使用率，並顯示每一台機器目前的使用率在一個表格中'
];

const PromptInput: React.FC<PromptInputProps> = ({
  onSubmit,
  isLoading,
  error,
  validationErrors,
  initialPrompt = ''
}) => {
  const [prompt, setPrompt] = useState(initialPrompt);

  useEffect(() => {
    setPrompt(initialPrompt);
  }, [initialPrompt]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) {
      onSubmit(prompt.trim());
    }
  };

  const handleExampleClick = (example: string) => {
    setPrompt(example);
  };

  return (
    <div className="card max-w-3xl mx-auto">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        描述您想要的監控儀表板
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
            輸入自然語言描述
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="例如：顯示過去 1 小時的 CPU 使用率"
            className="input-field min-h-32 resize-y"
            disabled={isLoading}
          />
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
            <p className="font-medium">錯誤</p>
            <p className="text-sm">{error}</p>
            
            {/* 顯示詳細的 validation 錯誤 */}
            {validationErrors && validationErrors.length > 0 && (
              <div className="mt-3 space-y-2">
                <p className="text-sm font-medium">驗證錯誤詳情：</p>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  {validationErrors.map((validationError, index) => (
                    <li key={index} className="ml-2">
                      <span className="font-medium">{validationError.type}:</span>{' '}
                      {validationError.message}
                      {validationError.path && (
                        <span className="block ml-5 text-xs text-red-600 mt-1">
                          路徑: {validationError.path}
                        </span>
                      )}
                      {validationError.widget_id && (
                        <span className="block ml-5 text-xs text-red-600 mt-1">
                          Widget ID: {validationError.widget_id}
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={isLoading || !prompt.trim()}
            className="btn-primary flex-1"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                生成中...
              </span>
            ) : (
              '生成儀表板'
            )}
          </button>
        </div>
      </form>

      <div className="mt-8">
        <h3 className="text-sm font-medium text-gray-700 mb-3">
          快速範例：
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {EXAMPLE_PROMPTS.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              disabled={isLoading}
              className="text-left px-4 py-3 bg-gray-50 hover:bg-gray-100 rounded-lg text-sm text-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PromptInput;
