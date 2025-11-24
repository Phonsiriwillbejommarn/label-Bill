/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                glass: {
                    bg: 'rgba(255, 255, 255, 0.05)',
                    border: 'rgba(255, 255, 255, 0.1)',
                    shadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
                }
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            animation: {
                'pulse-glow': 'pulse-glow 2s infinite',
            },
            keyframes: {
                'pulse-glow': {
                    '0%, 100%': { boxShadow: '0 0 5px rgba(0, 242, 254, 0.2)' },
                    '50%': { boxShadow: '0 0 20px rgba(0, 242, 254, 0.6)' },
                }
            }
        },
    },
    plugins: [],
}
