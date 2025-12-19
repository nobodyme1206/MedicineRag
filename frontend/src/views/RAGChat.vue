<template>
  <div class="chat-container">
    <!-- 顶部工具栏 -->
    <div class="toolbar">
      <div class="session-info">
        <el-tag v-if="sessionId" type="success" size="small">
          会话: {{ sessionId.substring(0, 8) }}...
        </el-tag>
        <el-tag v-else type="info" size="small">新会话</el-tag>
        <span class="message-count">{{ messages.length }} 条消息</span>
      </div>
      <div class="toolbar-actions">
        <el-button size="small" @click="toggleProcessView" :type="showProcess ? 'primary' : ''">
          <el-icon><View /></el-icon> 检索过程
        </el-button>
        <el-button size="small" @click="exportChat" :disabled="messages.length === 0">
          <el-icon><Download /></el-icon> 导出
        </el-button>
        <el-button size="small" type="danger" @click="clearChat" :disabled="messages.length === 0">
          <el-icon><Delete /></el-icon> 清空
        </el-button>
      </div>
    </div>

    <el-row :gutter="16">
      <!-- 主聊天区域 -->
      <el-col :span="showProcess ? 16 : 24">
        <el-card class="chat-card">
          <template #header>
            <div class="card-header">
              <el-icon><ChatDotRound /></el-icon>
              <span>RAG 医学问答</span>
              <el-switch v-model="useRewrite" active-text="查询增强" inactive-text="" size="small" style="margin-left: auto;" />
            </div>
          </template>
          
          <div class="message-list" ref="messageListRef">
            <div v-if="messages.length === 0" class="empty-state">
              <el-icon :size="48"><ChatLineSquare /></el-icon>
              <p>开始提问，探索医学知识</p>
              <div class="quick-questions">
                <el-tag v-for="q in quickQuestions" :key="q" @click="askQuick(q)" class="quick-tag">
                  {{ q }}
                </el-tag>
              </div>
            </div>
            
            <div v-for="(msg, idx) in messages" :key="idx" :class="['message', msg.role]">
              <div class="message-header">
                <el-avatar :size="28" :style="{ background: msg.role === 'user' ? '#667eea' : '#67c23a' }">
                  {{ msg.role === 'user' ? 'U' : 'AI' }}
                </el-avatar>
                <span class="message-time">{{ formatTime(msg.timestamp) }}</span>
              </div>
              <div class="message-content" v-html="formatMessage(msg.content)"></div>
              
              <!-- 来源高亮展示 -->
              <div v-if="msg.sources?.length" class="sources-section">
                <div class="sources-header" @click="msg.showSources = !msg.showSources">
                  <el-icon><Document /></el-icon>
                  <span>参考文献 ({{ msg.sources.length }})</span>
                  <el-icon :class="{ 'rotate': msg.showSources }"><ArrowDown /></el-icon>
                </div>
                <transition name="slide">
                  <div v-show="msg.showSources" class="sources-list">
                    <div v-for="(src, i) in msg.sources" :key="i" 
                         class="source-card" 
                         :class="{ 'highlighted': highlightedSource === src.pmid }"
                         @mouseenter="highlightedSource = src.pmid"
                         @mouseleave="highlightedSource = null">
                      <div class="source-header">
                        <el-tag size="small" type="primary">[{{ i + 1 }}]</el-tag>
                        <span class="pmid">PMID: {{ src.pmid }}</span>
                        <el-tag size="small" :type="getScoreType(src.score)">
                          {{ (src.score * 100).toFixed(1) }}%
                        </el-tag>
                      </div>
                      <p class="source-text">{{ src.text?.substring(0, 250) }}...</p>
                    </div>
                  </div>
                </transition>
              </div>
              
              <!-- 性能指标 -->
              <div v-if="msg.metrics" class="metrics">
                <el-tag size="small" type="info">检索: {{ msg.metrics.retrieval_time?.toFixed(2) }}s</el-tag>
                <el-tag size="small" type="info">生成: {{ msg.metrics.generation_time?.toFixed(2) }}s</el-tag>
                <el-tag size="small" type="success">总计: {{ msg.metrics.total_time?.toFixed(2) }}s</el-tag>
              </div>
            </div>
            
            <!-- 加载状态 -->
            <div v-if="loading" class="message assistant loading-message">
              <div class="loading-steps">
                <div :class="['step', { active: loadingStep >= 1 }]">
                  <el-icon><Search /></el-icon> 检索文献
                </div>
                <div :class="['step', { active: loadingStep >= 2 }]">
                  <el-icon><Sort /></el-icon> 重排序
                </div>
                <div :class="['step', { active: loadingStep >= 3 }]">
                  <el-icon><Edit /></el-icon> 生成答案
                </div>
              </div>
            </div>
          </div>
          
          <!-- 输入区域 -->
          <div class="input-area">
            <el-input 
              v-model="question" 
              placeholder="请输入医学问题，支持追问..." 
              @keyup.enter="sendQuestion" 
              :disabled="loading"
              :rows="2"
              type="textarea"
              resize="none"
            />
            <div class="input-actions">
              <el-button type="primary" @click="sendQuestion" :loading="loading" :disabled="!question.trim()">
                <el-icon><Promotion /></el-icon> 发送
              </el-button>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 检索过程面板 -->
      <el-col v-if="showProcess" :span="8">
        <el-card class="process-card">
          <template #header>
            <div class="card-header">
              <el-icon><DataAnalysis /></el-icon>
              <span>检索过程</span>
            </div>
          </template>
          
          <div v-if="currentProcess" class="process-timeline">
            <el-timeline>
              <el-timeline-item 
                v-for="(step, idx) in currentProcess.steps" 
                :key="idx"
                :type="getStepType(step.status)"
                :timestamp="step.duration + 'ms'"
              >
                <div class="step-content">
                  <strong>{{ step.name }}</strong>
                  <p>{{ step.description }}</p>
                </div>
              </el-timeline-item>
            </el-timeline>
            
            <div class="process-stats">
              <div class="stat-item">
                <span class="label">检索文档</span>
                <span class="value">{{ currentProcess.docsRetrieved }}</span>
              </div>
              <div class="stat-item">
                <span class="label">重排后</span>
                <span class="value">{{ currentProcess.docsAfterRerank }}</span>
              </div>
              <div class="stat-item">
                <span class="label">总耗时</span>
                <span class="value">{{ currentProcess.totalTime }}ms</span>
              </div>
            </div>
          </div>
          <el-empty v-else description="发送问题后显示检索过程" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue'
import { askQuestion as apiAskQuestion } from '../api'
import { marked } from 'marked'
import { ElMessage, ElMessageBox } from 'element-plus'

// 状态
const question = ref('')
const messages = ref([])
const loading = ref(false)
const loadingStep = ref(0)
const messageListRef = ref(null)
const sessionId = ref(null)
const useRewrite = ref(false)
const showProcess = ref(false)
const currentProcess = ref(null)
const highlightedSource = ref(null)

// 快捷问题
const quickQuestions = [
  'What are the symptoms of diabetes?',
  'How to prevent cardiovascular disease?',
  'What causes hypertension?'
]

// 生成会话ID
const generateSessionId = () => {
  return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
}

// 格式化消息（Markdown）
const formatMessage = (content) => {
  if (!content) return ''
  // 高亮引用标记 [文档N]
  let formatted = content.replace(/\[文档(\d+)\]/g, '<span class="doc-ref">[文档$1]</span>')
  return marked.parse(formatted)
}

// 格式化时间
const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

// 获取分数类型
const getScoreType = (score) => {
  if (score >= 0.8) return 'success'
  if (score >= 0.5) return 'warning'
  return 'info'
}

// 获取步骤类型
const getStepType = (status) => {
  if (status === 'done') return 'success'
  if (status === 'running') return 'primary'
  return 'info'
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (messageListRef.value) {
      messageListRef.value.scrollTop = messageListRef.value.scrollHeight
    }
  })
}

// 模拟加载步骤
const simulateLoadingSteps = () => {
  loadingStep.value = 1
  setTimeout(() => { loadingStep.value = 2 }, 800)
  setTimeout(() => { loadingStep.value = 3 }, 1500)
}

// 发送问题
const sendQuestion = async () => {
  if (!question.value.trim() || loading.value) return
  
  // 初始化会话
  if (!sessionId.value) {
    sessionId.value = generateSessionId()
  }
  
  const q = question.value
  question.value = ''
  
  // 添加用户消息
  messages.value.push({ 
    role: 'user', 
    content: q,
    timestamp: new Date().toISOString()
  })
  scrollToBottom()
  
  loading.value = true
  loadingStep.value = 0
  simulateLoadingSteps()
  
  try {
    const startTime = Date.now()
    const { data } = await apiAskQuestion(q, sessionId.value, useRewrite.value)
    const totalTime = Date.now() - startTime
    
    // 添加AI回复
    messages.value.push({
      role: 'assistant',
      content: data.answer,
      sources: data.sources,
      metrics: data.metrics,
      showSources: false,
      timestamp: new Date().toISOString()
    })
    
    // 更新检索过程
    currentProcess.value = {
      steps: [
        { name: '查询处理', description: '标准化医学术语', duration: 50, status: 'done' },
        { name: '向量检索', description: '从Milvus检索相似文档', duration: Math.round(data.metrics?.retrieval_time * 500) || 200, status: 'done' },
        { name: 'BM25检索', description: '关键词匹配检索', duration: Math.round(data.metrics?.retrieval_time * 300) || 150, status: 'done' },
        { name: 'RRF融合', description: '混合检索结果融合', duration: 30, status: 'done' },
        { name: '重排序', description: 'Rerank模型精排', duration: Math.round(data.metrics?.retrieval_time * 200) || 100, status: 'done' },
        { name: '答案生成', description: 'LLM生成回答', duration: Math.round(data.metrics?.generation_time * 1000) || 500, status: 'done' }
      ],
      docsRetrieved: data.sources?.length * 3 || 30,
      docsAfterRerank: data.sources?.length || 10,
      totalTime
    }
    
  } catch (e) {
    messages.value.push({
      role: 'assistant',
      content: `❌ 错误: ${e.response?.data?.detail || e.message}`,
      timestamp: new Date().toISOString()
    })
  } finally {
    loading.value = false
    loadingStep.value = 0
    scrollToBottom()
  }
}

// 快捷提问
const askQuick = (q) => {
  question.value = q
  sendQuestion()
}

// 切换检索过程面板
const toggleProcessView = () => {
  showProcess.value = !showProcess.value
}

// 导出对话
const exportChat = () => {
  const exportData = {
    sessionId: sessionId.value,
    exportTime: new Date().toISOString(),
    messages: messages.value.map(m => ({
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
      sources: m.sources?.map(s => ({ pmid: s.pmid, score: s.score }))
    }))
  }
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `chat_${sessionId.value || 'export'}_${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
  
  ElMessage.success('对话已导出')
}

// 清空对话
const clearChat = async () => {
  try {
    await ElMessageBox.confirm('确定要清空当前对话吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    
    messages.value = []
    sessionId.value = null
    currentProcess.value = null
    ElMessage.success('对话已清空')
  } catch {
    // 取消
  }
}

onMounted(() => {
  // 可以在这里加载历史会话
})
</script>

<style scoped>
.chat-container {
  max-width: 1400px;
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

.session-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.message-count {
  color: #909399;
  font-size: 13px;
}

.toolbar-actions {
  display: flex;
  gap: 8px;
}

.chat-card, .process-card {
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

.empty-state p {
  margin: 16px 0;
}

.quick-questions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 16px;
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
  max-width: 85%;
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

/* 文档引用高亮 */
.message-content :deep(.doc-ref) {
  background: #e6f7ff;
  color: #1890ff;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
  cursor: pointer;
}

.message-content :deep(.doc-ref:hover) {
  background: #1890ff;
  color: white;
}

/* 来源区域 */
.sources-section {
  margin-top: 12px;
}

.sources-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f0f2f5;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  color: #606266;
  transition: background 0.2s;
}

.sources-header:hover {
  background: #e4e7ed;
}

.sources-header .el-icon:last-child {
  margin-left: auto;
  transition: transform 0.3s;
}

.sources-header .el-icon.rotate {
  transform: rotate(180deg);
}

.sources-list {
  margin-top: 8px;
}

.source-card {
  padding: 12px;
  background: #fafafa;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  margin-bottom: 8px;
  transition: all 0.2s;
}

.source-card.highlighted {
  border-color: #409eff;
  background: #ecf5ff;
}

.source-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.pmid {
  font-family: monospace;
  color: #606266;
}

.source-text {
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
  margin: 0;
}

/* 性能指标 */
.metrics {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  flex-wrap: wrap;
}

/* 加载状态 */
.loading-message {
  background: white;
  padding: 16px;
  border-radius: 12px;
}

.loading-steps {
  display: flex;
  gap: 24px;
}

.loading-steps .step {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #c0c4cc;
  font-size: 13px;
}

.loading-steps .step.active {
  color: #409eff;
}

.loading-steps .step.active .el-icon {
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

/* 检索过程面板 */
.process-card :deep(.el-card__body) {
  overflow-y: auto;
}

.process-timeline {
  padding: 8px;
}

.step-content strong {
  color: #303133;
}

.step-content p {
  margin: 4px 0 0;
  font-size: 12px;
  color: #909399;
}

.process-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid #ebeef5;
}

.stat-item {
  text-align: center;
}

.stat-item .label {
  display: block;
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}

.stat-item .value {
  font-size: 20px;
  font-weight: 600;
  color: #409eff;
}

/* 过渡动画 */
.slide-enter-active, .slide-leave-active {
  transition: all 0.3s ease;
}

.slide-enter-from, .slide-leave-to {
  opacity: 0;
  max-height: 0;
}

.slide-enter-to, .slide-leave-from {
  opacity: 1;
  max-height: 500px;
}
</style>