import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import type { UISpec, WidgetCodePatch } from '@ui-agent/types';
import UIRenderer from '../components/UIRenderer';
import { compileWidgetCode } from '../services/widget-compiler';
import { widgetRegistry } from '../services/widget-registry';
import { fetchDashboardData, fetchDashboardMeta } from '../services/dashboard-api';

function applyWidgetPatches(widgetPatches: WidgetCodePatch[]) {
  for (const patch of widgetPatches) {
    if (patch.compile_success && patch.generated_file_path) {
      console.debug(
        `[widget-patch] ${patch.widget_type} 已由後端落地並編譯：${patch.generated_file_path}`
      );
      continue;
    }

    try {
      console.debug(
        `[widget-patch] compiling ${patch.widget_type}, preview:`,
        (patch.code || '').slice(0, 200)
      );
      const component = compileWidgetCode(patch);
      widgetRegistry.registerCustom(patch.widget_type, component);
    } catch (compileErr) {
      console.error(`Failed to compile/register widget patch: ${patch.widget_type}`, compileErr);
    }
  }
}

function DashboardPage() {
  const navigate = useNavigate();
  const { dashboardId } = useParams<{ dashboardId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const initialVersion = useMemo(() => searchParams.get('version') ?? '', [searchParams]);

  const [uiSpec, setUISpec] = useState<UISpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCheckingUpdate, setIsCheckingUpdate] = useState(false);
  const [versionToken, setVersionToken] = useState<string>(initialVersion);
  const [updatedAt, setUpdatedAt] = useState<string>('');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const loadDashboard = useCallback(
    async (targetId: string) => {
      setIsLoading(true);
      setError(null);
      setStatusMessage(null);
      try {
        const data = await fetchDashboardData(targetId);
        const patches = (data.widgetPatches || []) as WidgetCodePatch[];
        if (patches.length > 0) {
          applyWidgetPatches(patches);
        }
        setUISpec(data.uiSpec);
        setVersionToken(data.versionToken);
        setUpdatedAt(data.updatedAt);
        setSearchParams({ version: data.versionToken });
      } catch (err: unknown) {
        const data = err as { message?: string };
        setError(data.message || 'Failed to load dashboard');
      } finally {
        setIsLoading(false);
      }
    },
    [setSearchParams]
  );

  const handleCheckUpdates = useCallback(async () => {
    if (!dashboardId) return;
    setIsCheckingUpdate(true);
    setStatusMessage(null);
    try {
      const meta = await fetchDashboardMeta(dashboardId);
      if (meta.versionToken !== versionToken) {
        await loadDashboard(dashboardId);
        setStatusMessage('已載入最新版本。');
      } else {
        setUpdatedAt(meta.updatedAt);
        setStatusMessage('目前已是最新版本。');
      }
    } catch (err: unknown) {
      const data = err as { message?: string };
      setStatusMessage(data.message || '檢查更新失敗。');
    } finally {
      setIsCheckingUpdate(false);
    }
  }, [dashboardId, loadDashboard, versionToken]);

  useEffect(() => {
    if (!dashboardId) {
      setError('Dashboard ID is required');
      setIsLoading(false);
      return;
    }
    void loadDashboard(dashboardId);
  }, [dashboardId, loadDashboard]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-4 flex flex-wrap gap-3 items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-gray-800">監控儀表板</h2>
            <p className="text-sm text-gray-600">Dashboard ID: {dashboardId}</p>
            {updatedAt && <p className="text-xs text-gray-500">最後更新：{updatedAt}</p>}
            {versionToken && <p className="text-xs text-gray-500">版本：{versionToken}</p>}
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              className="btn-secondary"
              disabled={isCheckingUpdate || isLoading}
              onClick={() => void handleCheckUpdates()}
            >
              {isCheckingUpdate ? '檢查中...' : '檢查更新'}
            </button>
            <button type="button" className="btn-secondary" onClick={() => navigate('/')}>重新生成</button>
          </div>
        </div>

        {statusMessage && (
          <div className="mb-4 rounded-md border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-700">
            {statusMessage}
          </div>
        )}

        {isLoading ? (
          <div className="card">載入儀表板中...</div>
        ) : error ? (
          <div className="card">
            <p className="text-red-600 mb-4">{error}</p>
            <Link to="/" className="btn-secondary inline-block">
              返回生成頁
            </Link>
          </div>
        ) : uiSpec ? (
          <UIRenderer uiSpec={uiSpec} />
        ) : (
          <div className="card">尚無可用儀表板資料。</div>
        )}
      </div>
    </div>
  );
}

export default DashboardPage;
