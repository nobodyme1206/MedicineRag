import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 统一错误处理
api.interceptors.response.use(
  response => response,
  error => {
    const message = error.response?.data?.detail || error.message || '请求失败'
    console.error('API Error:', message)
    return Promise.reject(error)
  }
)

// RAG 问答（支持会话）
export const askQuestion = (question, sessionId = null, useRewrite = false) => {
  return api.post('/v1/ask', {
    question,
    session_id: sessionId,
    use_rewrite: useRewrite
  })
}

// 批量问答
export const askBatch = (questions, useSemanticCache = true) => {
  return api.post('/v1/ask/batch', {
    questions,
    use_semantic_cache: useSemanticCache
  })
}

// Agent 问答 - 统一命名，同时导出两个名称保持兼容
export const agentQuery = (query, maxSteps = 5, verbose = true) => {
  return api.post('/v1/agent', {
    query,
    max_steps: maxSteps,
    verbose
  })
}

// 别名，供 AgentChat.vue 使用
export const agentChat = agentQuery

// 文献检索
export const searchLiterature = (query, topK = 10, method = 'hybrid') => {
  return api.post('/v1/retrieve', {
    query,
    top_k: topK,
    method
  })
}

// 健康检查
export const healthCheck = () => {
  return api.get('/health')
}

// 系统统计
export const getStats = () => {
  return api.get('/v1/stats')
}

// 会话管理
export const getSessionHistory = (sessionId) => {
  return api.get(`/v1/session/${sessionId}/history`)
}

export const clearSession = (sessionId) => {
  return api.delete(`/v1/session/${sessionId}`)
}

// 缓存管理
export const getCacheStats = () => {
  return api.get('/v1/cache/stats')
}

export const prewarmCache = (queries = null) => {
  return api.post('/v1/cache/prewarm', queries)
}

export const clearCache = (cacheType = 'all') => {
  return api.delete(`/v1/cache/clear?cache_type=${cacheType}`)
}

export default api
