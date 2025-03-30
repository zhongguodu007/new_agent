<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Lock, User } from '@element-plus/icons-vue'
import { useUserStore } from '../stores/user'

const router = useRouter()
const userStore = useUserStore()
const isLogin = ref(true) // true表示登录模式，false表示注册模式

// 登录表单数据
const loginForm = reactive({
  username: '',
  password: ''
})

// 注册表单数据
const registerForm = reactive({
  username: '',
  password: '',
  confirmPassword: ''
})

// 表单校验规则
const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const registerRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 20, message: '用户名长度应为3-20个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度最少为6个字符', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    {
      validator: (rule, value, callback) => {
        if (value !== registerForm.password) {
          callback(new Error('两次输入的密码不一致'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ]
}

const loginFormRef = ref()
const registerFormRef = ref()

// 切换登录/注册模式
const toggleMode = () => {
  isLogin.value = !isLogin.value
}

// 处理登录
const handleLogin = async () => {
  if (!loginFormRef.value) return
  
  await loginFormRef.value.validate((valid, fields) => {
    if (valid) {
      // 使用Pinia进行登录验证
      if (userStore.login(loginForm.username, loginForm.password)) {
        ElMessage.success('登录成功')
        // 跳转到首页
        router.push('/')
      } else {
        ElMessage.error('用户名或密码错误')
      }
    }
  })
}

// 处理注册
const handleRegister = async () => {
  if (!registerFormRef.value) return
  
  await registerFormRef.value.validate((valid, fields) => {
    if (valid) {
      // 使用Pinia进行注册
      if (userStore.register(registerForm.username, registerForm.password)) {
        ElMessage.success('注册成功，请使用新账号登录')
        // 注册成功后切换到登录模式
        isLogin.value = true
        // 将注册的用户名自动填入登录表单
        loginForm.username = registerForm.username
        loginForm.password = ''
        
        // 清空注册表单
        registerForm.username = ''
        registerForm.password = ''
        registerForm.confirmPassword = ''
      }
    }
  })
}

// 在组件挂载时初始化用户状态
onMounted(() => {
  userStore.initUserState()
})
</script>

<template>
  <div class="login-container">
    <div class="login-box">
      <div class="login-header">
        <h2>{{ isLogin ? '用户登录' : '用户注册' }}</h2>
        <div class="tech-line"></div>
      </div>
      
      <!-- 登录表单 -->
      <div v-if="isLogin" class="form-container">
        <el-form
          ref="loginFormRef"
          :model="loginForm"
          :rules="loginRules"
          label-position="top"
        >
          <el-form-item prop="username" label="用户名">
            <el-input 
              v-model="loginForm.username"
              :prefix-icon="User"
              placeholder="请输入用户名"
            />
          </el-form-item>
          <el-form-item prop="password" label="密码">
            <el-input 
              v-model="loginForm.password"
              type="password"
              :prefix-icon="Lock"
              placeholder="请输入密码"
              show-password
            />
          </el-form-item>
          <div class="form-actions">
            <el-button type="primary" @click="handleLogin" class="submit-btn">登录</el-button>
          </div>
          <div class="form-footer">
            <p>没有账号？<a href="javascript:;" @click="toggleMode">立即注册</a></p>
          </div>
        </el-form>
      </div>

      <!-- 注册表单 -->
      <div v-else class="form-container">
        <el-form
          ref="registerFormRef"
          :model="registerForm"
          :rules="registerRules"
          label-position="top"
        >
          <el-form-item prop="username" label="用户名">
            <el-input 
              v-model="registerForm.username"
              :prefix-icon="User"
              placeholder="请输入3-20个字符的用户名"
            />
          </el-form-item>
          <el-form-item prop="password" label="密码">
            <el-input 
              v-model="registerForm.password"
              type="password"
              :prefix-icon="Lock"
              placeholder="请输入至少6个字符的密码"
              show-password
            />
          </el-form-item>
          <el-form-item prop="confirmPassword" label="确认密码">
            <el-input 
              v-model="registerForm.confirmPassword"
              type="password"
              :prefix-icon="Lock"
              placeholder="请再次输入密码"
              show-password
            />
          </el-form-item>
          <div class="form-actions">
            <el-button type="primary" @click="handleRegister" class="submit-btn">注册</el-button>
          </div>
          <div class="form-footer">
            <p>已有账号？<a href="javascript:;" @click="toggleMode">返回登录</a></p>
          </div>
        </el-form>
      </div>
    </div>
    
    <!-- 科技感装饰元素 -->
    <div class="tech-circles">
      <div class="circle circle-1"></div>
      <div class="circle circle-2"></div>
      <div class="circle circle-3"></div>
    </div>
    <div class="tech-grid"></div>
  </div>
</template>

<style scoped>
.login-container {
  min-height: calc(100vh - 60px);
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #1a2a6c, #12246c);
  position: relative;
  overflow: hidden;
}

.login-box {
  width: 450px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  padding: 40px;
  box-shadow: 0 15px 25px rgba(0, 0, 0, 0.2);
  position: relative;
  z-index: 10;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.25);
}

.login-header {
  text-align: center;
  margin-bottom: 30px;
}

.login-header h2 {
  color: #1a2a6c;
  font-size: 28px;
  margin-bottom: 15px;
}

.tech-line {
  height: 3px;
  background: linear-gradient(90deg, transparent, #409eff, transparent);
  margin: 0 auto;
  position: relative;
}

.tech-line:after {
  content: '';
  position: absolute;
  height: 100%;
  width: 30px;
  background: #409eff;
  left: 50%;
  transform: translateX(-50%);
  animation: pulse 2s infinite;
}

.form-container {
  margin-top: 20px;
}

.form-actions {
  margin-top: 25px;
}

.submit-btn {
  width: 100%;
  height: 50px;
  font-size: 16px;
  background: linear-gradient(90deg, #409eff, #007bff);
  border: none;
  letter-spacing: 1px;
  padding: 12px 0;
  transition: transform 0.3s;
}

.submit-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 123, 255, 0.3);
}

.form-footer {
  margin-top: 20px;
  text-align: center;
}

.form-footer a {
  color: #409eff;
  text-decoration: none;
  font-weight: 600;
}

.tech-circles {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
}

.circle {
  position: absolute;
  border-radius: 50%;
  border: 2px solid rgba(64, 158, 255, 0.2);
}

.circle-1 {
  width: 300px;
  height: 300px;
  bottom: -150px;
  left: -150px;
  border-width: 4px;
  border-color: rgba(64, 158, 255, 0.3);
  animation: rotate 20s linear infinite;
}

.circle-2 {
  width: 500px;
  height: 500px;
  top: -250px;
  right: -250px;
  border-width: 2px;
  border-color: rgba(0, 123, 255, 0.2);
  animation: rotate 25s linear infinite reverse;
}

.circle-3 {
  width: 200px;
  height: 200px;
  bottom: 50px;
  right: 100px;
  border-width: 3px;
  border-color: rgba(125, 200, 255, 0.2);
  animation: rotate 15s linear infinite;
}

.tech-grid {
  position: absolute;
  width: 200%;
  height: 200%;
  top: -50%;
  left: -50%;
  background-image: 
    linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px);
  background-size: 50px 50px;
  transform: perspective(500px) rotateX(60deg);
  animation: grid-move 15s linear infinite;
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@keyframes pulse {
  0% {
    opacity: 0.5;
  }
  50% {
    opacity: 1;
  }
  100% {
    opacity: 0.5;
  }
}

@keyframes grid-move {
  from {
    transform: perspective(500px) rotateX(60deg) translateY(0);
  }
  to {
    transform: perspective(500px) rotateX(60deg) translateY(50px);
  }
}

/* 响应式设计 */
@media (max-width: 576px) {
  .login-box {
    width: 90%;
    padding: 20px;
  }
}
</style>
