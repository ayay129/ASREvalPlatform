import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import NewEval from './pages/NewEval'
import Report from './pages/Report'
import Compare from './pages/Compare'
import TrainRuns from './pages/TrainRuns'
import NewTrainRun from './pages/NewTrainRun'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/train-runs" element={<TrainRuns />} />
          <Route path="/train-runs/new" element={<NewTrainRun />} />
          <Route path="/new" element={<NewEval />} />
          <Route path="/report/:id" element={<Report />} />
          <Route path="/compare" element={<Compare />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
