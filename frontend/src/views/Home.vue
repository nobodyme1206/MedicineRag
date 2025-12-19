<template>
  <div class="home">
    <div class="hero">
      <h2>基于RAG的智能医学文献检索与问答系统</h2>
      <p>融合150万+医学文献，提供精准的医学知识问答服务</p>
    </div>
    
    <el-row :gutter="20" class="features">
      <el-col :span="8">
        <el-card shadow="hover" @click="$router.push('/rag')">
          <template #header>
            <div class="card-header">
              <el-icon><ChatDotRound /></el-icon>
              <span>RAG问答</span>
            </div>
          </template>
          <p>基于检索增强生成技术，从医学文献中检索相关内容并生成准确答案</p>
        </el-card>
      </el-col>
      
      <el-col :span="8">
        <el-card shadow="hover" @click="$router.push('/agent')">
          <template #header>
            <div class="card-header">
              <el-icon><Cpu /></el-icon>
              <span>智能Agent</span>
            </div>
          </template>
          <p>Adaptive RAG智能代理，支持智能路由、查询分解、自我反思和检索增强</p>
        </el-card>
      </el-col>
      
      <el-col :span="8">
        <el-card shadow="hover" @click="$router.push('/search')">
          <template #header>
            <div class="card-header">
              <el-icon><Search /></el-icon>
              <span>文献检索</span>
            </div>
          </template>
          <p>支持BM25、向量检索和混合检索，快速定位相关医学文献</p>
        </el-card>
      </el-col>
    </el-row>
    
    <el-card class="stats-card" v-if="stats">
      <template #header>系统状态</template>
      <el-descriptions :column="3">
        <el-descriptions-item label="RAG状态">
          <el-tag :type="stats.rag_status === 'ready' ? 'success' : 'danger'">
            {{ stats.rag_status }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Agent状态">
          <el-tag :type="stats.agent_status === 'ready' ? 'success' : 'danger'">
            {{ stats.agent_status }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="索引文档数">
          {{ stats.documents_indexed?.toLocaleString() || 'N/A' }}
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getStats } from '../api'

const stats = ref(null)

onMounted(async () => {
  try {
    const { data } = await getStats()
    stats.value = data
  } catch (e) {
    console.error('获取状态失败', e)
  }
})
</script>

<style scoped>
.home {
  max-width: 1000px;
  margin: 0 auto;
}

.hero {
  text-align: center;
  padding: 60px 20px;
  color: white;
}

.hero h2 {
  font-size: 28px;
  margin-bottom: 16px;
}

.features {
  margin-bottom: 30px;
}

.features .el-card {
  cursor: pointer;
  transition: transform 0.3s;
}

.features .el-card:hover {
  transform: translateY(-5px);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.stats-card {
  margin-top: 20px;
}
</style>
