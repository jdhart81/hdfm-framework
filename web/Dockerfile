FROM node:22-bookworm-slim
WORKDIR /app
COPY package*.json ./
COPY vendor ./vendor
RUN npm ci --omit=dev
COPY server ./server
COPY public ./public
COPY drizzle ./drizzle
COPY db ./db
RUN mkdir .data && chown -R node:node .data
USER node
ENV HOST=0.0.0.0 PORT=4173
EXPOSE 4173
CMD ["node", "server/local.mjs"]
