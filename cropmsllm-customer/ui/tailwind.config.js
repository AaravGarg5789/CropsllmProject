/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        canvas: "#f8fafc",
        line: "#e2e8f0",
        muted: "#475569",
        seg: {
          bg: "#323232",
          maize: "#1ec83c",
          weed: "#dc2828",
          weedB: "#ff8c00",
          weedC: "#ffdc14",
          weedD: "#b450dc",
        },
      },
      fontFamily: {
        sans: ["system-ui", "Arial", "Helvetica", "sans-serif"],
        mono: ["Consolas", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
