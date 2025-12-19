<template>
  <div class="chat-container">
    <!-- 顶部工具栏 -->
    <div class="toolbar">
      <div class="agent-info">
        <div class="agent-badge">
          <el-icon class="agent-icon"><Cpu /></el-icon>
          <span class="agent-name">智能医学助手</span>
        </div>
        <el-divider direction="vertical" />
        <span class="message-count">{{ messages.length }} 条对话</span>
      </div>
      <div class="toolbar-actions">
        <el-button size="small" @click="exportChat" :disabled="messages.length === 0">
          <el-icon><Download /></el-icon> 导出
        </el-button>
        <el-button size="small" type="danger" @click="clearChat" :disabled="messages.length === 0">
          <el-icon><Delete /></el-icon> 清空
        </el-button>
      </div>
    </div>

    <div class="chat-main">
      <!-- 消息列表 -->
      <div class="message-list" ref="messageListRef">
        <!-- 空状态 -->
        <div v-if="messages.length === 0" class="empty-state">
          <div class="empty-icon">
            <el-icon :size="56"><Cpu /></el-icon>
          </div>
          <h3>智能医学问答助手</h3>
          <p>基于 Adaptive RAG 架构，智能分析问题复杂度</p>
          <div class="feature-list">
            <div class="feature-item">
              <el-icon><Guide /></el-icon>
              <span>智能路由分析</span>
            </div>
            <div class="feature-item">
              <el-icon><Connection /></el-icon>
              <span>复杂问题自动分解</span>
            </div>
            <div class="feature-item">
              <el-icon><Refresh /></el-icon>
              <span>检索质量自我反思</span>
            </div>
          </div>
          <div class="quick-questions">
            <span class="quick-label">快速提问:</span>
            <el-button v-for="q in quickQuestions" :key="q" size="small" round @click="askQuick(q)">
              {{ q }}
            </el-button>
          </div>
        </div>
        
        <!-- 消息列表 -->
        <template v-for="(msg, idx) in messages" :key="idx">
          <!-- 用户消息 -->
          <div v-if="msg.role === 'user'" class="message-row user">
            <div class="message-bubble user">
              <div class="bubble-content">{{ msg.content }}</div>
              <div class="bubble-time">{{ formatTime(msg.timestamp) }}</div>
            </div>
          </div>
          
          <!-- AI回答 -->
          <div v-else class="message-row assistant">
            <div class="message-card">
              <!-- 回答内容 -->
              <div class="card-body" v-html="formatMessage(msg.content)"></div>
              
              <!-- 参考来源 -->
              <div v-if="msg.sources?.length" class="sources-section">
                <div class="sources-toggle" @click="msg.showSources = !msg.showSources">
                  <el-icon><Document /></el-icon>
                  <span>参考来源 ({{ msg.sources.length }})</span>
                  <el-icon class="toggle-icon" :class="{ expanded: msg.showSources }"><ArrowDown /></el-icon>
                </div>
                <el-collapse-transition>
                  <div v-show="msg.showSources" class="sources-content">
                    <div v-for="(src, i) in msg.sources" :key="i" class="source-item">
                      <span class="source-num">[{{ i + 1 }}]</span>
                      <span class="source-pmid">PMID: {{ src.pmid }}</span>
                      <el-progress 
                        :percentage="Math.round(src.score * 100)" 
                        :stroke-width="6"
                        :color="getScoreColor(src.score)"
                        style="flex: 1; margin: 0 12px;"
                      />
                      <span class="source-score">{{ (src.score * 100).toFixed(0) }}%</span>
                    </div>
                  </div>
                </el-collapse-transition>
              </div>

              <!-- 底部指标 -->
              <div v-if="msg.metrics" class="card-footer">
                <el-tag v-if="msg.complexity" size="small" :type="getComplexityType(msg.complexity)">
                  {{ getComplexityLabel(msg.complexity) }}
                </el-tag>
                <span class="metric-divider">|</span>
                <span class="metric">检索 {{ msg.metrics.retrievalTime?.toFixed(2) || '0.00' }}s</span>
                <span class="metric-divider">|</span>
                <span class="metric">生成 {{ msg.metrics.generationTime?.toFixed(2) || '0.00' }}s</span>
                <span class="metric-divider">|</span>
                <span class="metric highlight">共 {{ msg.metrics.totalTime?.toFixed(2) || '0.00' }}s</span>
              </div>
            </div>
          </div>
        </template>
        
        <!-- 加载状态 -->
        <div v-if="loading" class="message-row assistant">
          <div class="message-card loading">
            <div class="loading-bar">
              <div class="loading-step" :class="{ active: loadingPhase >= 1, done: loadingPhase > 1 }">
                <el-icon><Guide /></el-icon>
                <span>分析</span>
              </div>
              <div class="loading-line" :class="{ active: loadingPhase >= 2 }"></div>
              <div class="loading-step" :class="{ active: loadingPhase >= 2, done: loadingPhase > 2 }">
                <el-icon><Search /></el-icon>
                <span>检索</span>
              </div>
              <div class="loading-line" :class="{ active: loadingPhase >= 3 }"></div>
              <div class="loading-step" :class="{ active: loadingPhase >= 3, done: loadingPhase > 3 }">
                <el-icon><View /></el-icon>
                <span>反思</span>
              </div>
              <div class="loading-line" :class="{ active: loadingPhase >= 4 }"></div>
              <div class="loading-step" :class="{ active: loadingPhase >= 4 }">
                <el-icon><Edit /></el-icon>
                <span>生成</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 输入区域 -->
      <div class="input-area">
        <el-input 
          v-model="question" 
          placeholder="请输入医学问题..." 
          @keyup.enter="sendQuestion" 
          :disabled="loading"
          size="large"
        >
          <template #append>
            <el-button type="primary" @click="sendQuestion" :loading="loading">
              发送
            </el-button>
          </template>
        </el-input>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, computed } from 'vue'
import { agentChat } from '../api'
import { marked } from 'marked'
import { ElMessage, ElMessageBox } from 'element-plus'

const question = ref('')
const messages = ref([])
const loading = ref(false)
const loadingPhase = ref(0)
const messageListRef = ref(null)

const quickQuestions = [
  '什么是糖尿病？',
  '高血压的症状有哪些？',
  '如何预防心血管疾病？'
]

const formatMessage = (content) => {
  if (!content) return ''
  // 将 [文档N] 转换为带样式的上标引用
  let formatted = content.replace(/\[文档(\d+)\]/g, '<sup class="doc-ref">[$1]</sup>')
  return marked.parse(formatted)
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  return new Date(timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

const getComplexityType = (complexity) => {
  if (complexity === 'simple') return 'success'
  if (complexity === 'moderate') return 'warning'
  return 'danger'
}

const getComplexityLabel = (complexity) => {
  const labels = { simple: '简单', moderate: '中等', complex: '复杂' }
  return labels[complexity] || complexity
}

const getScoreColor = (score) => {
  if (score >= 0.8) return '#67c23a'
  if (score >= 0.5) return '#e6a23c'
  return '#909399'
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messageListRef.value) {
      messageListRef.value.scrollTop = messageListRef.value.scrollHeight
    }
  })
}

const simulateLoading = () => {
  loadingPhase.value = 1
  setTimeout(() => { loadingPhase.value = 2 }, 500)
  setTimeout(() => { loadingPhase.value = 3 }, 1200)
  setTimeout(() => { loadingPhase.value = 4 }, 1800)
}

const sendQuestion = async () => {
  if (!question.value.trim() || loading.value) return
  
  const q = question.value
  question.value = ''
  messages.value.push({ 
    role: 'user', 
    content: q,
    timestamp: new Date().toISOString()
  })
  scrollToBottom()
  
  loading.value = true
  simulateLoading()
  
  try {
    const startTime = Date.now()
    const { data } = await agentChat(q)
    const totalTime = (Date.now() - startTime) / 1000
    
    messages.value.push({
      role: 'assistant',
      content: data.answer,
      sources: data.sources,
      complexity: data.complexity,
      showSources: false,
      metrics: {
        retrievalTime: data.metrics?.retrieval_time || 0,
        generationTime: data.metrics?.generation_time || 0,
        totalTime: totalTime
      },
      timestamp: new Date().toISOString()
    })
  } catch (e) {
    messages.value.push({
      role: 'assistant',
      content: `❌ 错误: ${e.response?.data?.detail || e.message}`,
      timestamp: new Date().toISOString()
    })
  } finally {
    loading.value = false
    loadingPhase.value = 0
    scrollToBottom()
  }
}

const askQuick = (q) => {
  question.value = q
  sendQuestion()
}

const exportChat = () => {
  const exportData = {
    exportTime: new Date().toISOString(),
    messages: messages.value
  }
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `agent_chat_${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
  ElMessage.success('对话已导出')
}

const clearChat = async () => {
  try {
    await ElMessageBox.confirm('确定要清空当前对话吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    messages.value = []
    ElMessage.success('对话已清空')
  } catch {}
}
</script>

<style scoped>
.chat-container {
  max-width: 900px;
  margin: 0 auto;
  padding: 16px;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 顶部工具栏 */
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  margin-bottom: 16px;
}

.agent-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.agent-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 24px;
  color: white;
  font-weight: 500;
  font-size: 14px;
}

.message-count {
  color: #909399;
  font-size: 13px;
}

.toolbar-actions {
  display: flex;
  gap: 8px;
}

/* 聊天区域 */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  overflow: hidden;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  background: #f8f9fa;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #606266;
}

.empty-icon {
  width: 100px;
  height: 100px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  margin-bottom: 20px;
}

.empty-state h3 {
  margin: 0 0 8px;
  color: #303133;
  font-size: 20px;
}

.empty-state p {
  margin: 0 0 24px;
  color: #909399;
  font-size: 14px;
}

.feature-list {
  display: flex;
  gap: 32px;
  margin-bottom: 32px;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #606266;
  font-size: 13px;
}

.feature-item .el-icon {
  color: #667eea;
}

.quick-questions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: center;
}

.quick-label {
  color: #909399;
  font-size: 13px;
}

/* 消息行 */
.message-row {
  margin-bottom: 20px;
}

.message-row.user {
  display: flex;
  justify-content: flex-end;
}

/* 用户消息气泡 */
.message-bubble.user {
  max-width: 70%;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 12px 16px;
  border-radius: 16px 16px 4px 16px;
}

.bubble-content {
  line-height: 1.6;
  word-break: break-word;
}

.bubble-time {
  font-size: 11px;
  opacity: 0.7;
  margin-top: 6px;
  text-align: right;
}

/* AI消息卡片 */
.message-card {
  background: white;
  border-radius: 12px;
  border-left: 4px solid #67c23a;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  overflow: hidden;
}

.message-card .card-body {
  padding: 16px 20px;
  line-height: 1.8;
  color: #303133;
}

.card-body :deep(p) {
  margin: 0 0 12px;
}

.card-body :deep(p:last-child) {
  margin-bottom: 0;
}

.card-body :deep(ul), .card-body :deep(ol) {
  margin: 12px 0;
  padding-left: 20px;
}

.card-body :deep(li) {
  margin-bottom: 8px;
}

.card-body :deep(strong) {
  color: #303133;
}

/* 文档引用上标 */
.card-body :deep(.doc-ref) {
  color: #667eea;
  font-weight: 600;
  font-size: 11px;
  cursor: pointer;
  margin: 0 1px;
}

.card-body :deep(.doc-ref:hover) {
  color: #764ba2;
  text-decoration: underline;
}

/* 参考来源 */
.sources-section {
  border-top: 1px solid #f0f0f0;
}

.sources-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  cursor: pointer;
  color: #606266;
  font-size: 13px;
  transition: background 0.2s;
}

.sources-toggle:hover {
  background: #f5f7fa;
}

.toggle-icon {
  margin-left: auto;
  transition: transform 0.3s;
}

.toggle-icon.expanded {
  transform: rotate(180deg);
}

.sources-content {
  padding: 0 20px 16px;
}

.source-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-bottom: 8px;
  font-size: 13px;
}

.source-item:last-child {
  margin-bottom: 0;
}

.source-num {
  color: #667eea;
  font-weight: 600;
}

.source-pmid {
  color: #606266;
  font-family: monospace;
}

.source-score {
  color: #67c23a;
  font-weight: 500;
  min-width: 40px;
  text-align: right;
}

/* 底部指标 */
.card-footer {
  padding: 10px 20px;
  border-top: 1px solid #f0f0f0;
  font-size: 12px;
  color: #909399;
  display: flex;
  align-items: center;
  gap: 8px;
  background: #fafafa;
}

.metric-divider {
  color: #dcdfe6;
}

.metric.highlight {
  color: #67c23a;
  font-weight: 500;
}

/* 加载状态 */
.message-card.loading {
  padding: 20px;
}

.loading-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
}

.loading-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  color: #c0c4cc;
  font-size: 12px;
  padding: 0 16px;
}

.loading-step .el-icon {
  font-size: 20px;
}

.loading-step.active {
  color: #667eea;
}

.loading-step.active .el-icon {
  animation: bounce 0.6s infinite;
}

.loading-step.done {
  color: #67c23a;
}

.loading-line {
  width: 40px;
  height: 2px;
  background: #e4e7ed;
  border-radius: 1px;
  margin: 0 4px;
}

.loading-line.active {
  background: #67c23a;
}

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

/* 输入区域 */
.input-area {
  padding: 16px;
  border-top: 1px solid #f0f0f0;
  background: white;
}

.input-area :deep(.el-input-group__append) {
  background: #667eea;
  border-color: #667eea;
}

.input-area :deep(.el-input-group__append .el-button) {
  color: white;
  border-color: #667eea;
  background: #667eea;
}

.input-area :deep(.el-input-group__append .el-button:hover) {
  background: #764ba2;
  border-color: #764ba2;
}
</style>
