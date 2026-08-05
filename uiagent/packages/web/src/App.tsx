import { Navigate, Route, Routes } from 'react-router-dom';
import GeneratePage from './pages/GeneratePage';
import DashboardPage from './pages/DashboardPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<GeneratePage />} />
      <Route path="/dashboards/:dashboardId" element={<DashboardPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
