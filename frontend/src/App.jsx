import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Evaluations from './pages/Evaluations'
import NewEval from './pages/NewEval'
import Report from './pages/Report'
import Compare from './pages/Compare'
import TrainRuns from './pages/TrainRuns'
import NewTrainRun from './pages/NewTrainRun'
import TrainRunDetail from './pages/TrainRunDetail'
import Datasets from './pages/Datasets'
import Models from './pages/Models'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/train-runs" element={<TrainRuns />} />
          <Route path="/train-runs/new" element={<NewTrainRun />} />
          <Route path="/train-runs/:id" element={<TrainRunDetail />} />
          <Route path="/models" element={<Models />} />
          <Route path="/evaluations" element={<Evaluations />} />
          <Route path="/new" element={<NewEval />} />
          <Route path="/report/:id" element={<Report />} />
          <Route path="/compare" element={<Compare />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
