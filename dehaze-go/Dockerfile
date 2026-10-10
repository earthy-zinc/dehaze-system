# dehaze-go 镜像
#
# 两个构建目标，构建上下文都是「仓库根」：单测要读仓库根 .env，而 config.GoRoot() 依赖
# go.mod 锚定，也要求 dehaze-go/ 位于仓库根下一级的原始目录结构。
#
#   # 测试镜像（三端单测回归用：自带源码 + 预热好的模块缓存，Pod 内直接 go test ./...）
#   docker build -f dehaze-go/Dockerfile --target test -t csighub.tencentyun.com/earthywu/dehaze-system:go-test-<sha> .
#   # 服务镜像（默认目标）
#   docker build -f dehaze-go/Dockerfile -t dehaze-go:latest .
#
# 运行容器时用环境变量注入配置（config.Init 读 RepoRoot()/.env，文件不存在时忽略）。

# ---------- 测试镜像：源码 + 工具链 ----------
FROM golang:1.25 AS test

ENV GOTOOLCHAIN=local \
    GOPATH=/go \
    GOCACHE=/root/.cache/go-build \
    GOPROXY=https://goproxy.cn,direct \
    GOSUMDB=off \
    TZ=Asia/Shanghai

# 容器内布局与宿主检出路径无关：/app/dehaze-go + /app/.env（Secret 挂载，config.GoRoot() 靠目录结构定位）
WORKDIR /app/dehaze-go

# 模块缓存在构建期预热：运行期不依赖 goproxy 可达性
COPY dehaze-go/go.mod dehaze-go/go.sum ./
RUN go mod download

COPY dehaze-go /app/dehaze-go
# 测试库由 testutil 用仓库根 config/sql 全量重建（与 java/python 同一套脚本），缺它所有建库用例必失败
COPY config /app/config

CMD ["go", "test", "./..."]

# ---------- 编译 ----------
FROM test AS build
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/dehaze-go ./cmd

# ---------- 服务镜像 ----------
FROM alpine:3.20
RUN sed -i 's@dl-cdn.alpinelinux.org@mirrors.cloud.tencent.com@g' /etc/apk/repositories \
    && apk --no-cache add tzdata ca-certificates \
    && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone
WORKDIR /app
COPY --from=build /out/dehaze-go ./dehaze-go
COPY dehaze-go/config ./config
EXPOSE 8990
CMD ["./dehaze-go"]
