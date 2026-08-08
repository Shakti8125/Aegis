import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        aegis: {
          bg: "#0B0E14",
          surface: "#131822",
          card: "#1A202C",
          border: "#232B3E",
          hover: "#2D374D",
          healthy: "#3DDC97",
          degraded: "#F5A623",
          critical: "#E5484D",
          muted: "#7C89A3",
          accent: "#38BDF8",
          text: "#F1F5F9",
        },
      },
      fontFamily: {
        sans: ["Inter", "IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "JetBrains Mono", "Menlo", "monospace"],
      },
      animation: {
        "pulse-fast": "pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "glow-ping": "ping 1.5s cubic-bezier(0, 0, 0.2, 1) 2",
        "trace-flow": "traceFlow 2s linear infinite",
      },
      keyframes: {
        traceFlow: {
          "0%": { strokeDashoffset: "24" },
          "100%": { strokeDashoffset: "0" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
