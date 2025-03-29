<script setup>
import { RouterLink, RouterView } from 'vue-router'
import { ref } from 'vue'

const activeIndex = ref('home')
</script>

<template>
  <div class="app-container">
    <header>
      <div class="logo">专业领域智能体系统</div>
      <el-menu
        :default-active="activeIndex"
        class="nav-menu"
        mode="horizontal"
        router
      >
        <el-menu-item index="/" route="/">首页</el-menu-item>
        <el-menu-item index="/chat" route="/chat">智能对话</el-menu-item>
        <el-menu-item index="/knowledge-base" route="/knowledge-base">知识库管理</el-menu-item>
      </el-menu>
    </header>

    <main>
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

header {
  display: flex;
  align-items: center;
  padding: 0 20px;
  height: 60px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  position: relative;
  z-index: 100;
}

.logo {
  font-size: 20px;
  font-weight: bold;
  margin-right: 40px;
  color: #409eff;
}

.nav-menu {
  border-bottom: none;
}

main {
  flex: 1;
  padding: 0;
  background-color: #f5f7fa;
}
</style>
