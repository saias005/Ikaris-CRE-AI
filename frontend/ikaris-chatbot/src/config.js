// frontend/src/config.js
const config = {
  // For unified deployment, use relative URL (empty string)
  // This makes API calls go to the same origin
  API_URL: process.env.REACT_APP_API_URL || ''
};

export default config;