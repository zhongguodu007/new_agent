import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('../views/ChatView.vue'),
    },
    {
      path: '/knowledge-base',
      name: 'knowledgeBase',
      component: () => import('../views/KnowledgeBaseView.vue'),
    }
  ],
})

export default router
