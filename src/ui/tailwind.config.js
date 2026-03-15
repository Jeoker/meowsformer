/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,vue}",
  ],
  theme: {
    extend: {
      colors: {
        meow: {
          50:  "#fff9f0",
          100: "#fef2e0",
          200: "#fde4c0",
          300: "#fcd09a",
          400: "#f9b46a",
          500: "#f79535",
          600: "#e07720",
          700: "#bc5e16",
          800: "#984c17",
          900: "#7c3f16",
        },
      },
    },
  },
  plugins: [],
}
