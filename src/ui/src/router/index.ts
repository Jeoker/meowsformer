import { createRouter, createWebHistory } from "vue-router";
import TranslatePage from "../pages/TranslatePage.vue";
import AboutPage from "../pages/AboutPage.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "demo", component: TranslatePage },
    { path: "/about", name: "about", component: AboutPage },
  ],
  scrollBehavior() {
    return { top: 0 };
  },
});
