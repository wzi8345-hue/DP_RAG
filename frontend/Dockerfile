# DP-RAG 前端镜像：pnpm 构建 → nginx 静态托管（SPA history 回退）。
FROM node:22-slim AS build
WORKDIR /app
RUN corepack enable

# 构建期注入（CI 传入）：后端地址与 Logto 配置
ARG VITE_API_BASE=https://funmg.dp.tech/sci-loop-api
ARG VITE_LOGTO_ENDPOINT=https://auth.dplink.cc/
ARG VITE_LOGTO_APP_ID=skjc9b4p12ykvz40vshjc
ARG VITE_LOGTO_RESOURCE=https://funmg.dp.tech/sci-loop-api
ARG VITE_LOGTO_SCOPES=all:data
ENV VITE_API_BASE=$VITE_API_BASE \
    VITE_LOGTO_ENDPOINT=$VITE_LOGTO_ENDPOINT \
    VITE_LOGTO_APP_ID=$VITE_LOGTO_APP_ID \
    VITE_LOGTO_RESOURCE=$VITE_LOGTO_RESOURCE \
    VITE_LOGTO_SCOPES=$VITE_LOGTO_SCOPES

COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile || pnpm install
COPY . .
RUN pnpm build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
