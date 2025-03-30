import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  // 状态
  state: () => ({
    username: '',
    isLoggedIn: false,
  }),
  
  // getter
  getters: {
    // 获取用户登录状态
    getUserLoginStatus: (state) => state.isLoggedIn,
    // 获取用户名
    getUsername: (state) => state.username,
  },
  
  // actions
  actions: {
    // 登录
    login(username, password) {
      // 模拟验证：默认用户usr1，密码123
      if (username === 'usr1' && password === '123') {
        this.isLoggedIn = true
        this.username = username
        
        // 存储登录状态到本地存储
        localStorage.setItem('userInfo', JSON.stringify({
          username: this.username,
          isLoggedIn: true
        }))
        
        return true
      }
      return false
    },
    
    // 注册(模拟)
    register(username, password) {
      // 实际项目中会发送请求到后端进行注册
      // 这里只是模拟成功
      return true
    },
    
    // 退出登录
    logout() {
      this.isLoggedIn = false
      this.username = ''
      
      // 清除本地存储
      localStorage.removeItem('userInfo')
    },
    
    // 初始化用户状态(从本地存储加载)
    initUserState() {
      const userInfo = JSON.parse(localStorage.getItem('userInfo') || '{"isLoggedIn": false}')
      if (userInfo.isLoggedIn) {
        this.isLoggedIn = true
        this.username = userInfo.username
      }
    }
  }
}) 