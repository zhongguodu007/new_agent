<script setup>
import { RouterLink, RouterView } from 'vue-router'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from './stores/user'
import UserInfo from './components/UserInfo.vue'

const router = useRouter()
const userStore = useUserStore()
const activeIndex = ref('home')

// 在组件挂载时初始化用户状态
onMounted(() => {
  userStore.initUserState()
})
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
      
      <div class="user-area">
        <UserInfo />
        <el-button v-if="!userStore.isLoggedIn" type="primary" @click="router.push('/login')">登录</el-button>
      </div>
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
  background-color: #1a2a6c;
}

.logo {
  font-size: 20px;
  font-weight: bold;
  margin-right: 40px;
  color: #fff;
}

.nav-menu {
  border-bottom: none;
  flex-grow: 1;
  background-color: transparent;
}

:deep(.el-menu--horizontal .el-menu-item) {
  color: #fff;
}

:deep(.el-menu--horizontal .el-menu-item.is-active) {
  color: #409eff;
  background-color: rgba(255, 255, 255, 0.1);
}

.user-area {
  display: flex;
  align-items: center;
  gap: 10px;
}

main {
  flex: 1;
  padding: 0;
  background-color: #f5f7fa;
}
</style>
