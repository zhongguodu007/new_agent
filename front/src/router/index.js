import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: { requiresAuth: true }
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('../views/ChatView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/knowledge-base',
      name: 'knowledgeBase',
      component: () => import('../views/KnowledgeBaseView.vue'),
      meta: { requiresAuth: true }
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/UserLogin.vue'),
    }
  ],
})

// 全局前置守卫
router.beforeEach((to, from, next) => {
  // 检查用户是否需要登录
  if (to.matched.some(record => record.meta.requiresAuth)) {
    // 检查用户是否已登录
    const userInfo = JSON.parse(localStorage.getItem('userInfo') || '{"isLoggedIn": false}')
    if (!userInfo.isLoggedIn) {
      // 未登录则重定向到登录页面
      next({ name: 'login' })
    } else {
      // 已登录则正常访问
      next()
    }
  } else {
    // 不需要登录验证的页面直接放行
    next()
  }
})

export default router
