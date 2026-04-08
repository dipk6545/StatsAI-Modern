module.exports = {
  apps: [
    {
      name: 'stats-ai-backend',
      script: 'python',
      args: 'main.py',
      cwd: './server',
      watch: true,
      env: {
        PORT: 3001,
      },
    },
    {
      name: 'stats-ai-frontend-nicegui',
      script: 'python',
      args: 'main.py',
      cwd: './nicegui_app',
      watch: true,
    },
  ],
};
