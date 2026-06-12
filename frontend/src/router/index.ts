import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/chat' },
  {
    path: '/chat',
    name: 'chat',
    component: () => import('@/pages/ChatPage.vue'),
  },
  {
    path: '/library',
    name: 'library',
    component: () => import('@/pages/LibraryPage.vue'),
  },
  {
    path: '/skills',
    name: 'skills',
    component: () => import('@/pages/SkillsPage.vue'),
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/pages/SettingsPage.vue'),
  },
  {
    path: '/admin',
    name: 'admin',
    component: () => import('@/pages/AdminPage.vue'),
  },
  {
    path: '/callback',
    name: 'callback',
    component: () => import('@/pages/CallbackPage.vue'),
    meta: { public: true },
  },
  {
    path: '/s/:token',
    name: 'shared-conversation',
    component: () => import('@/pages/SharedConversationPage.vue'),
    meta: { public: true },
  },
  { path: '/:pathMatch(.*)*', redirect: '/chat' },
]

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})
