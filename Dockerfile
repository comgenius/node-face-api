# Syntax: docker/dockerfile:1

# Use Node LTS on Debian slim for maximum compatibility with native modules (tfjs-node, canvas)
FROM node:18-slim AS base

# Set workdir
WORKDIR /app

# Install system dependencies required for canvas and potential native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    python3 \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    libpixman-1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install production deps without using package-lock to avoid stale transitive versions
COPY package.json ./
RUN npm install --omit=dev --no-package-lock

# Copy the rest of the application
COPY . .

# Expose app port (configurable via PORT env)
EXPOSE 3000

# Default environment
ENV NODE_ENV=production \
    PORT=3000

# Optional healthcheck hitting the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Start server
CMD ["npm", "start"]
