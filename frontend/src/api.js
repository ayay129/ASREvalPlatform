import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

export const api = {
  // Datasets
  listDatasets: () =>
    http.get('/datasets').then(r => r.data),

  // Train runs
  listTrainRuns: (params = {}) =>
    http.get('/train-runs', { params }).then(r => r.data),

  createTrainRun: (body) =>
    http.post('/train-runs', body).then(r => r.data),

  getTrainRun: (id) =>
    http.get(`/train-runs/${id}`).then(r => r.data),

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
