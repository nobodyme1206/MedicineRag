import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/Home.vue')
  },
  {
    path: '/rag',
    name: 'RAG',
    component: () => import('../views/RAGChat.vue')
  },
  {
    path: '/agent',
    name: 'Agent',
    component: () => import('../views/AgentChat.vue')
  },
  {
    path: '/search',
    name: 'Search',
    component: () => import('../views/LiteratureSearch.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
