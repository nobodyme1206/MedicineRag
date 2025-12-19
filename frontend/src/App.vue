<template>
  <el-container class="app-container">
    <el-header>
      <div class="header-content">
        <h1>🏥 医学知识问答系统</h1>
        <el-menu mode="horizontal" :router="true" :default-active="$route.path">
          <el-menu-item index="/">首页</el-menu-item>
          <el-menu-item index="/rag">RAG问答</el-menu-item>
          <el-menu-item index="/agent">智能Agent</el-menu-item>
          <el-menu-item index="/search">文献检索</el-menu-item>
        </el-menu>
      </div>
    </el-header>
    <el-main>
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup>
</script>

<style scoped>
.app-container {
  min-height: 100vh;
}

.el-header {
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  padding: 0 20px;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
}

.header-content h1 {
  font-size: 20px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.el-main {
  padding: 20px;
}
</style>
