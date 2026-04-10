import { createRouter, createWebHistory } from "vue-router";
import TranslatePage from "../pages/TranslatePage.vue";
import AboutPage from "../pages/AboutPage.vue";
import PrivacyPage from "../pages/PrivacyPage.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "demo", component: TranslatePage },
    { path: "/about", name: "about", component: AboutPage },
    { path: "/privacy", name: "privacy", component: PrivacyPage },
  ],
  scrollBehavior() {
    return { top: 0 };
  },
});
