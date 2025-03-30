<script setup>
import { useUserStore } from '../stores/user'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

const userStore = useUserStore()
const router = useRouter()

// 处理退出登录
const handleLogout = () => {
  userStore.logout()
  ElMessage.success('已退出登录')
  router.push('/login')
}
</script>

<template>
  <div v-if="userStore.isLoggedIn" class="user-info">
    <span class="welcome">欢迎您，{{ userStore.username }}</span>
    <el-button type="danger" size="small" @click="handleLogout">退出登录</el-button>
  </div>
</template>

<style scoped>
.user-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.welcome {
  font-size: 14px;
  color: #fff;
}
</style> 