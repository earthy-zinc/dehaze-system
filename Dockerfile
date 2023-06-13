FROM node:16-alpine as build-stage

WORKDIR /note

COPY package.json package-lock.json ./
RUN npm config set registry http://registry.npmmirror.org && \
    npm install

COPY . .
ARG NODE_ENV=""
RUN env ${NODE_ENV} npm docs:build

## -- stage: dist => nginx --
FROM nginx:alpine

ENV TZ=Asia/Shanghai

COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage /note/docs/.vuepress/dist /usr/share/nginx/html

EXPOSE 80
