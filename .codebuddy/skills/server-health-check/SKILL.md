---
name: server-health-check
description: 服务器健康检查与清理优化技能。当用户提到"服务器分析"、"系统检查"、"清理进程"、"磁盘清理"、"内存优化"、"端口检查"、"Docker清理"、"RPM包分析"、"僵尸进程"、"冗余服务"等关键词时使用此技能。适用于 Linux 服务器（含各云厂商及物理机）、macOS 开发机和 Windows 系统的全面深度诊断、冗余资源识别与清理优化。
---

# 服务器健康检查与清理优化

## 目的

对目标机器进行全面深度诊断，覆盖系统资源、软件包、容器、进程、端口、磁盘等维度，识别冗余/垃圾资源并提供可执行的清理方案。支持 Linux（RPM/DEB/APK）、macOS 和 Windows。

## 工作流程

### 采集

根据目标系统执行对应的采集脚本：

- **Linux / macOS**: `bash ${SKILL_DIR}/scripts/collect.sh`
- **Windows**: `powershell -ExecutionPolicy Bypass -File ${SKILL_DIR}/scripts/collect.ps1`

脚本会自动探测操作系统类型、包管理器、容器运行时、云环境，并输出结构化数据。若脚本执行失败，根据报错信息修复后重试，不要手动逐项采集。

### 分析诊断

基于采集数据，按以下维度逐项分析：

**软件包** -- 将已安装包分为：核心系统（禁删）、容器运行时（按需）、开发工具链（非编译场景可删）、旧内核（保留当前 + 前一版本）、调试符号（通常可删）、硬件固件（云/虚拟机可删，物理机谨慎）、GUI/桌面（纯 CLI 可删）。macOS 上分析 brew 安装的 formulae 和 cask，检查 brew cleanup 可回收空间。Windows 上分析注册表安装程序、winget/chocolatey/scoop 包，识别废弃软件。

**开发环境** -- 检测 Node.js(nvm/nvm-windows)、Python(pyenv/conda)、Java(SDKMAN)、Go 等多版本共存情况，评估每个版本的磁盘占用，建议仅保留活跃使用的版本。检查全局 npm/pip 包是否有废弃项。

**容器** -- 识别 Exited/Dead 容器、悬空镜像、同镜像重复容器、未使用 Volume、Build Cache。检查是否有缺少 `--rm` 导致的容器泄漏。

**进程** -- 按启动时间（`START` 列）分组，同类进程存在多个不同时间启动的实例时，仅保留最新一组，其余为历史遗留。重点关注 IDE 远程开发旧会话、MCP Server 重复实例、重复应用进程（java/python/node/go 等同名多实例）、僵尸进程（Linux/macOS）。云厂商 Agent 进程标记为受保护，默认不清理。

**端口** -- 以 `0.0.0.0` / `:::` 绑定的数据库端口（3306/5432/27017/6379 等）为高危，对象存储/管理面板端口（9000/9090/8080 等）为中危。

**磁盘** -- 分区使用率 >70% 需关注，>90% 紧急。检查大目录归属、日志膨胀、缓存目录（`/var/log/`、`~/.cache/`、`node_modules/`、Docker overlay 存储）。macOS 关注 `~/Library/Caches`、Xcode DerivedData、ASL 日志、DiagnosticReports。Windows 关注 `%TEMP%`、`%LOCALAPPDATA%`、nuget/npm/pip 缓存、WSL ext4.vhdx 虚拟磁盘膨胀、Windows 事件日志空间。

**防火墙** -- Linux 检查 firewalld/ufw/iptables 状态和规则。macOS 检查应用防火墙和 pf 状态。Windows 检查各 Profile 的防火墙启用状态和默认策略。

### 输出报告

按以下结构以表格形式输出：

- **系统概况** -- CPU/内存/磁盘/Swap 一览
- **容器分析** -- 业务容器 vs 冗余容器，可回收空间
- **进程分析** -- TOP 资源占用 + 冗余/可清理进程
- **软件包分析** -- 可删除包分类及预估回收空间
- **端口安全** -- 对外暴露端口及风险等级
- **磁盘热点** -- 大目录/日志/缓存可回收空间
- **优化建议** -- 按 P0（零风险立即执行）/ P1（低风险近期执行）/ P2（可选优化）分级

### 执行清理

- **P0（零风险）**: 删除 Exited 容器、杀历史遗留重复进程、清理悬空镜像和 Build Cache
- **P1（低风险）**:
  - Linux: 删除旧内核/debuginfo 包、清理日志（`journalctl --vacuum-size=500M`）、清理包管理器缓存（`yum clean all` / `apt clean`）
  - macOS: `brew cleanup`、清理 Xcode DerivedData（`rm -rf ~/Library/Developer/Xcode/DerivedData`）、清理 ASL 日志（`sudo rm /var/log/asl/*.asl`）
  - Windows: 清理事件日志（`wevtutil cl Application`）、清理 TEMP（`Remove-Item $env:TEMP\* -Recurse -Force`）、磁盘清理（`cleanmgr /sagerun:1`）
- **P2（可选）**: 删除非必要开发工具链和多余版本、清理用户级缓存（npm/pip/gradle/nuget/pnpm store）、未使用 Docker Volume、配置日志轮转（Linux: logrotate, macOS: newsyslog, Windows: 事件日志大小限制）、收缩 WSL ext4.vhdx（`wsl --shutdown && Optimize-VHD`）

## 红线规则

- **云厂商 Agent 默认禁止清理** -- 删除会导致监控/安全/运维功能失效
- **内核清理始终保留当前版本和至少一个回退版本**
- **物理服务器不可删除硬件固件包**
- **删除软件包前必须检查反向依赖** -- RPM 用 `rpm -e --test`，DEB 用 `apt-get -s remove`，macOS 用 `brew deps --installed`，Windows 用 `winget` 确认无依赖
- **macOS 不可删除 /System、SIP 保护的文件**
- **Windows 不可删除 WinSxS 组件存储或修改 TrustedInstaller 所有权文件**
- **开发环境多版本清理前确认无项目依赖该版本**（检查 .nvmrc / .python-version / .java-version 等）
- **所有破坏性操作执行前向用户确认**
