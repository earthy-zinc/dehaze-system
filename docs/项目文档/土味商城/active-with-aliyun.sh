#!/bin/bash

# 前端80-86 88 后端90-94 105-106
ports=(
85 # earthy-delivery
90 # kob
91 # dehaze-java
92 # dehaze-python
93 # pei-blog
94 # tuwei-mall
105 # alist
106 # code-server
)

for port in "${ports[@]}"
do
  # 检查是否存在指定的 SSH 进程
  if ! pgrep -f "ssh -o ServerAliveInterval=60 -CNf -R ${port}:localhost:${port} root@47.120.48.158" > /dev/null; then
      # 如果不存在，执行 SSH 命令
      echo "$(date) 端口${port}已失效，重新执行代理命令"
      ssh -o ServerAliveInterval=60 -CNf -R "${port}":localhost:"${port}" root@47.120.48.158
  fi
done
