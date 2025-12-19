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

    <el-card class="chat-card">
      <template #header>
        <div class="card-header">
          <el-icon><Cpu /></el-icon>
          <span>智能 Agent 问答</span>
          <div class="header-tags">
            <el-tag size="small" type="info">智能路由</el-tag>
            <el-tag size="small" type="warning">查询分解</el-tag>
            <el-tag size="small" type="success">自我反思</el-tag>
          </div>
        </div>
      </template>
      
      <div class="message-list" ref="messageListRef">
        <div v-if="messages.length === 0" class="empty-state">
          <el-icon :size="48"><Cpu /></el-icon>
          <p>Adaptive RAG Agent 会智能分析问题复杂度</p>
          <div class="agent-features">
            <div class="feature">
              <el-icon><Guide /></el-icon>
              <span>简单问题快速回答</span>
            </div>
            <div class="feature">
              <el-icon><Connection /></el-icon>
              <span>复杂问题自动分解</span>
            </div>
            <div class="feature">
              <el-icon><Refresh /></el-icon>
              <span>检索质量自我反思</span>
            </div>
          </div>
          <div class="quick-questions">
            <el-tag v-for="q in quickQuestions" :key="q" @click="askQuick(q)" class="quick-tag">
              {{ q }}
            </el-tag>
          </div>
        </div>
        
        <div v-for="(msg, idx) in messages" :key="idx" :class="['message', msg.role]">
          <div class="message-header">
            <el-avatar :size="32" :style="{ background: msg.role === 'user' ? '#667eea' : '#67c23a' }">
              {{ msg.role === 'user' ? 'U' : 'AI' }}
            </el-avatar>
            <span class="message-time">{{ formatTime(msg.timestamp) }}</span>
            <el-tag v-if="msg.complexity" size="small" :type="getComplexityType(msg.complexity)">
              {{ msg.complexity }}
            </el-tag>
          </div>
          
          <div class="message-content" v-html="formatMessage(msg.content)"></div>
          
          <!-- Agent执行步骤可视化 -->
          <div v-if="msg.steps?.length" class="steps-section">
            <div class="steps-header" @click="msg.showSteps = !msg.showSteps">
              <el-icon><Operation /></el-icon>
              <span>执行步骤 ({{ msg.steps.length }})</span>
              <el-tag size="small" type="success">{{ msg.totalTime }}ms</el-tag>
              <el-icon :class="{ 'rotate': msg.showSteps }"><ArrowDown /></el-icon>
            </div>
            <transition name="slide">
              <div v-show="msg.showSteps" class="steps-timeline">
                <el-timeline>
                  <el-timeline-item 
                    v-for="(step, i) in msg.steps" 
                    :key="i"
                    :type="getStepColor(step.type)"
                    :timestamp="step.duration_ms + 'ms'"
                  >
                    <div class="step-card">
                      <div class="step-type">
                        <el-icon><component :is="getStepIcon(step.type)" /></el-icon>
                        {{ getStepName(step.type) }}
                      </div>
                    </div>
                  </el-timeline-item>
                </el-timeline>
              </div>
            </transition>
          </div>
          
          <!-- 来源信息 -->
          <div v-if="msg.sources?.length" class="sources-section">
            <div class="sources-header" @click="msg.showSources = !msg.showSources">
              <el-icon><Document /></el-icon>
              <span>参考来源 ({{ msg.sources.length }})</span>
              <el-icon :class="{ 'rotate': msg.showSources }"><ArrowDown /></el-icon>
            </div>
            <transition name="slide">
              <div v-show="msg.showSources" class="sources-list">
                <div v-for="(src, i) in msg.sources" :key="i" class="source-item">
                  <el-tag size="small" type="primary">[{{ i + 1 }}]</el-tag>
                  <span class="pmid">PMID: {{ src.pmid }}</span>
                  <el-tag size="small" :type="getScoreType(src.score)">
                    {{ (src.score * 100).toFixed(1) }}%
                  </el-tag>
                </div>
              </div>
            </transition>
          </div>
          
          <!-- 性能指标 -->
          <div v-if="msg.metrics" class="metrics">
            <el-tag size="small" type="info">相关性: {{ msg.metrics.relevance?.toFixed(1) }}</el-tag>
            <el-tag size="small" type="info">来源数: {{ msg.metrics.numSources }}</el-tag>
            <el-tag size="small" type="success">耗时: {{ msg.metrics.totalTime }}ms</el-tag>
          </div>
        </div>
        
        <!-- 加载状态 -->
        <div v-if="loading" class="message assistant loading-message">
          <div class="loading-animation">
            <div class="loading-step" :class="{ active: loadingPhase === 'route' }">
              <el-icon><Guide /></el-icon> 分析问题复杂度
            </div>
            <div class="loading-step" :class="{ active: loadingPhase === 'retrieve' }">
              <el-icon><Search /></el-icon> 检索相关文献
            </div>
            <div class="loading-step" :class="{ active: loadingPhase === 'reflect' }">
              <el-icon><View /></el-icon> 评估检索质量
            </div>
            <div class="loading-step" :class="{ active: loadingPhase === 'generate' }">
              <el-icon><Edit /></el-icon> 生成答案
            </div>
          </div>
        </div>
      </div>
      
      <!-- 输入区域 -->
      <div class="input-area">
        <el-input 
          v-model="question" 
          placeholder="输入医学问题，Agent会智能分析并回答..." 
          @keyup.enter="sendQuestion" 
          :disabled="loading"
          :rows="2"
          type="textarea"
          resize="none"
        />
        <div class="input-actions">
          <el-button type="primary" @click="sendQuestion" :loading="loading" :disabled="!question.trim()">
            <el-icon><Promotion /></el-icon> 执行
          </el-button>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { agentQuery } from '../api'
import { marked } from 'marked'
import { ElMessage, ElMessageBox } from 'element-plus'

const question = ref('')
const messages = ref([])
const loading = ref(false)
const loadingPhase = ref('')
const messageListRef = ref(null)

const quickQuestions = [
  '什么是糖尿病？',
  '1型和2型糖尿病有什么区别？',
  'How to prevent cardiovascular disease?'
]

const formatMessage = (content) => {
  if (!content) return ''
  let formatted = content.replace(/\[文档(\d+)\]/g, '<span class="doc-ref">[文档$1]</span>')
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

const getScoreType = (score) => {
  if (score >= 0.8) return 'success'
  if (score >= 0.5) return 'warning'
  return 'info'
}

const getStepColor = (type) => {
  const colors = {
    route: 'primary',
    decompose: 'warning',
    retrieve: 'success',
    reflect: 'info',
    generate: 'success'
  }
  return colors[type] || 'info'
}

const getStepIcon = (type) => {
  const icons = {
    route: 'Guide',
    decompose: 'Connection',
    retrieve: 'Search',
    reflect: 'View',
    generate: 'Edit'
  }
  return icons[type] || 'More'
}

const getStepName = (type) => {
  const names = {
    route: '路由判断',
    decompose: '查询分解',
    retrieve: '文献检索',
    reflect: '质量反思',
    generate: '答案生成'
  }
  return names[type] || type
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messageListRef.value) {
      messageListRef.value.scrollTop = messageListRef.value.scrollHeight
    }
  })
}

const simulateLoading = () => {
  loadingPhase.value = 'route'
  setTimeout(() => { loadingPhase.value = 'retrieve' }, 600)
  setTimeout(() => { loadingPhase.value = 'reflect' }, 1200)
  setTimeout(() => { loadingPhase.value = 'generate' }, 1800)
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
    const { data } = await agentQuery(q, 5, true)
    messages.value.push({
      role: 'assistant',
      content: data.answer,
      steps: data.steps,
      sources: data.sources,
      complexity: data.complexity,
      showSteps: false,
      showSources: false,
      totalTime: Math.round(data.total_time_ms),
      metrics: {
        relevance: data.relevance_score,
        numSources: data.num_sources,
        totalTime: Math.round(data.total_time_ms)
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
    loadingPhase.value = ''
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
    agentType: 'Adaptive RAG Agent',
    messages: messages.value.map(m => ({
      role: m.role,
      content: m.content,
      complexity: m.complexity,
      steps: m.steps,
      timestamp: m.timestamp
    }))
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
  max-width: 1000px;
  margin: 0 auto;
  padding: 16px;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding: 12px 16px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
  padding: 6px 14px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  color: white;
  font-weight: 500;
}

.agent-icon {
  font-size: 16px;
}

.agent-name {
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

.chat-card {
  height: calc(100vh - 180px);
  display: flex;
  flex-direction: column;
}

.chat-card :deep(.el-card__body) {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 0;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.header-tags {
  margin-left: auto;
  display: flex;
  gap: 6px;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: #f5f7fa;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}

.agent-features {
  display: flex;
  gap: 24px;
  margin: 20px 0;
}

.feature {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}

.quick-questions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}

.quick-tag {
  cursor: pointer;
  transition: all 0.2s;
}

.quick-tag:hover {
  transform: scale(1.05);
  background: #409eff;
  color: white;
}

.message {
  margin-bottom: 20px;
  max-width: 90%;
}

.message.user {
  margin-left: auto;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.message-time {
  font-size: 12px;
  color: #909399;
}

.message-content {
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
}

.message.user .message-content {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.message.assistant .message-content {
  background: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.message-content :deep(.doc-ref) {
  background: #e6f7ff;
  color: #1890ff;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
}

/* 步骤区域 */
.steps-section, .sources-section {
  margin-top: 12px;
}

.steps-header, .sources-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f0f2f5;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  transition: background 0.2s;
}

.steps-header:hover, .sources-header:hover {
  background: #e4e7ed;
}

.steps-header .el-icon:last-child,
.sources-header .el-icon:last-child {
  margin-left: auto;
  transition: transform 0.3s;
}

.el-icon.rotate {
  transform: rotate(180deg);
}

.steps-timeline {
  padding: 12px;
  background: #fafafa;
  border-radius: 8px;
  margin-top: 8px;
}

.step-card {
  padding: 8px;
}

.step-type {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
}

.sources-list {
  margin-top: 8px;
}

.source-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #fafafa;
  border-radius: 6px;
  margin-bottom: 6px;
}

.pmid {
  font-family: monospace;
  color: #606266;
}

.metrics {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  flex-wrap: wrap;
}

/* 加载动画 */
.loading-message {
  background: white;
  padding: 16px;
  border-radius: 12px;
}

.loading-animation {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.loading-step {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #c0c4cc;
  font-size: 13px;
  padding: 8px 12px;
  border-radius: 6px;
  transition: all 0.3s;
}

.loading-step.active {
  color: #409eff;
  background: #ecf5ff;
}

.loading-step.active .el-icon {
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 输入区域 */
.input-area {
  padding: 16px;
  border-top: 1px solid #ebeef5;
  background: white;
}

.input-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 12px;
}

/* 过渡动画 */
.slide-enter-active, .slide-leave-active {
  transition: all 0.3s ease;
}

.slide-enter-from, .slide-leave-to {
  opacity: 0;
  max-height: 0;
}
</style>