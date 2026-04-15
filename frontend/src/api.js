import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

export const api = {
  // Datasets (registry)
  listDatasets: (params = {}) =>
    http.get('/datasets', { params }).then(r => r.data),

  scanDatasets: () =>
    http.post('/datasets/scan').then(r => r.data),

  previewDataset: (id, n = 5) =>
    http.get(`/datasets/${id}/preview`, { params: { n } }).then(r => r.data),

  deleteDataset: (id) =>
    http.delete(`/datasets/${id}`).then(r => r.data),

  // HF pull jobs
  listDatasetPulls: () =>
    http.get('/dataset-pulls').then(r => r.data),

  createDatasetPull: (body) =>
    http.post('/dataset-pulls', body).then(r => r.data),

  deleteDatasetPull: (id) =>
    http.delete(`/dataset-pulls/${id}`).then(r => r.data),

  // legacy 扫描接口（保留）
  listDatasetsLegacy: () =>
    http.get('/datasets/legacy').then(r => r.data),

  // Train runs
  listTrainRuns: (params = {}) =>
    http.get('/train-runs', { params }).then(r => r.data),

  createTrainRun: (body) =>
    http.post('/train-runs', body).then(r => r.data),

  getTrainRun: (id) =>
    http.get(`/train-runs/${id}`).then(r => r.data),

  getTrainRunLog: (id, tail = 500) =>
    http.get(`/train-runs/${id}/log`, { params: { tail } }).then(r => r.data),

  getTrainRunMetrics: (id) =>
    http.get(`/train-runs/${id}/metrics`).then(r => r.data),

  // Evaluations
  listEvals: (params = {}) =>
    http.get('/evaluations', { params }).then(r => r.data),

  createEval: (body) =>
    http.post('/evaluations', body).then(r => r.data),

  getEval: (id, params = {}) =>
    http.get(`/evaluations/${id}`, { params }).then(r => r.data),

  deleteEval: (id) =>
    http.delete(`/evaluations/${id}`).then(r => r.data),

  exportReport: (id) =>
    `${http.defaults.baseURL}/evaluations/${id}/export`,

  // Compare
  compare: (ids) =>
    http.post('/compare', { evaluation_ids: ids }).then(r => r.data),
}
