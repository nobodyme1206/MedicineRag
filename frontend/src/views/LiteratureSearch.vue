<template>
  <div class="search-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <el-icon><Search /></el-icon>
          <span>文献检索</span>
        </div>
      </template>
      
      <el-form :inline="true" class="search-form">
        <el-form-item label="检索词">
          <el-input v-model="query" placeholder="输入检索关键词" style="width: 300px"
                    @keyup.enter="search" />
        </el-form-item>
        <el-form-item label="检索方式">
          <el-select v-model="method" style="width: 120px">
            <el-option label="混合检索" value="hybrid" />
            <el-option label="向量检索" value="vector" />
            <el-option label="BM25" value="bm25" />
          </el-select>
        </el-form-item>
        <el-form-item label="返回数量">
          <el-input-number v-model="topK" :min="1" :max="50" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="search" :loading="loading">检索</el-button>
        </el-form-item>
      </el-form>
      
      <div v-if="results.length" class="results-info">
        <el-tag>找到 {{ total }} 条结果</el-tag>
        <el-tag type="info">耗时 {{ latency.toFixed(2) }} ms</el-tag>
      </div>
      
      <div class="results-list">
        <el-card v-for="(item, idx) in results" :key="idx" class="result-item" shadow="hover">
          <div class="result-header">
            <el-tag size="small">PMID: {{ item.pmid }}</el-tag>
            <el-tag size="small" type="success">相关度: {{ (item.score * 100).toFixed(1) }}%</el-tag>
          </div>
          <p class="result-text">{{ item.text }}</p>
        </el-card>
        
        <el-empty v-if="searched && !results.length" description="未找到相关文献" />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { searchLiterature } from '../api'

const query = ref('')
const method = ref('hybrid')
const topK = ref(10)
const results = ref([])
const total = ref(0)
const latency = ref(0)
const loading = ref(false)
const searched = ref(false)

const search = async () => {
  if (!query.value.trim()) return
  
  loading.value = true
  searched.value = true
  
  try {
    const { data } = await searchLiterature(query.value, topK.value, method.value)
    results.value = data.results
    total.value = data.total
    latency.value = data.latency_ms
  } catch (e) {
    console.error('检索失败', e)
    results.value = []
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.search-container {
  max-width: 1000px;
  margin: 0 auto;
}

.search-form {
  margin-bottom: 20px;
}

.results-info {
  margin-bottom: 16px;
  display: flex;
  gap: 12px;
}

.results-list {
  max-height: 60vh;
  overflow-y: auto;
}

.result-item {
  margin-bottom: 12px;
}

.result-header {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

.result-text {
  font-size: 14px;
  line-height: 1.6;
  color: #666;
}
</style>
