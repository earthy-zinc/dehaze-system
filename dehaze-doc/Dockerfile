# 使用官方的 Nginx 镜像作为基础镜像
FROM nginx:stable-alpine

# 将构建生成的静态文件复制到 Nginx 服务器的服务目录下
COPY docs/.vuepress/dist /usr/share/nginx/html

# 复制自定义的 Nginx 配置文件（如果需要）
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 启动 Nginx 服务器
CMD ["nginx", "-g", "daemon off;"]
