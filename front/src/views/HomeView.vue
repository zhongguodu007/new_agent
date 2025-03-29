<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const navigateTo = (path) => {
  router.push(path)
}
</script>

<template>
  <div class="home-container">
    <section class="hero">
      <h1>专业领域智能体系统</h1>
      <p class="subtitle">针对通用大模型在垂直领域中知识深度不足的问题，我们构建了面向专业领域的智能体系统</p>
      
      <div class="action-buttons">
        <el-button type="primary" size="large" @click="navigateTo('/chat')">开始对话</el-button>
        <el-button size="large" @click="navigateTo('/knowledge-base')">管理知识库</el-button>
      </div>
    </section>
    
    <section class="features">
      <div class="feature-card">
        <el-icon class="feature-icon"><el-icon-document /></el-icon>
        <h3>自定义知识库</h3>
        <p>创建和管理专业领域的知识集合，添加和删除文本资源</p>
      </div>
      
      <div class="feature-card">
        <el-icon class="feature-icon"><el-icon-chat-dot-round /></el-icon>
        <h3>智能对话</h3>
        <p>支持全网搜索、本地知识库搜索和计算机专业知识检索</p>
      </div>
      
      <div class="feature-card">
        <el-icon class="feature-icon"><el-icon-connection /></el-icon>
        <h3>Agent工具调用</h3>
        <p>通过智能代理调用多种工具，实现更精准的专业领域回答</p>
      </div>
    </section>
  </div>
</template>

<style scoped>
.home-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
}

.hero {
  text-align: center;
  padding: 60px 20px;
}

h1 {
  font-size: 3rem;
  margin-bottom: 20px;
  color: #303133;
}

.subtitle {
  font-size: 1.2rem;
  max-width: 800px;
  margin: 0 auto 40px;
  color: #606266;
  line-height: 1.6;
}

.action-buttons {
  display: flex;
  gap: 20px;
  justify-content: center;
  margin-top: 30px;
}

.features {
  display: flex;
  justify-content: space-between;
  margin-top: 80px;
  gap: 30px;
}

.feature-card {
  flex: 1;
  background: white;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  text-align: center;
}

.feature-icon {
  font-size: 3rem;
  margin-bottom: 20px;
  color: #409eff;
}

.feature-card h3 {
  margin: 15px 0;
  font-size: 1.5rem;
  color: #303133;
}

.feature-card p {
  color: #606266;
  line-height: 1.6;
}
</style>
