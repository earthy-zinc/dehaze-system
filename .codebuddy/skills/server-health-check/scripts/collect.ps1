#Requires -Version 5.1
# 服务器健康检查 - Windows 一键数据采集脚本
# 支持: Windows 10/11, Windows Server 2016+
# 用法: powershell -ExecutionPolicy Bypass -File collect.ps1

$ErrorActionPreference = "SilentlyContinue"

function Section($title) { Write-Output "`n====== $title ======" }

# ============================================================
# 环境探测
# ============================================================
Section "环境探测"
$os = Get-CimInstance Win32_OperatingSystem
Write-Output "系统: $($os.Caption) $($os.Version)"
Write-Output "架构: $($os.OSArchitecture)"
Write-Output "主机: $env:COMPUTERNAME"
Write-Output "启动时间: $($os.LastBootUpTime)"

# 虚拟化检测
$cs = Get-CimInstance Win32_ComputerSystem
Write-Output "制造商: $($cs.Manufacturer)"
Write-Output "型号: $($cs.Model)"
if ($cs.Model -match "Virtual|VMware|VirtualBox|Hyper-V|KVM|Xen|QEMU") {
    Write-Output "虚拟化: 是 ($($cs.Model))"
} else {
    Write-Output "虚拟化: 物理机或未识别"
}

# ============================================================
# 系统概况
# ============================================================
Section "系统概况"

Write-Output "--- 内存 ---"
$totalMem = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
$freeMem = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$usedMem = [math]::Round($totalMem - $freeMem, 1)
Write-Output "总内存: ${totalMem} GB | 已用: ${usedMem} GB | 可用: ${freeMem} GB"

Write-Output "--- 磁盘 ---"
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    $total = [math]::Round($_.Size / 1GB, 1)
    $free = [math]::Round($_.FreeSpace / 1GB, 1)
    $used = [math]::Round($total - $free, 1)
    $pct = if ($total -gt 0) { [math]::Round(($used / $total) * 100, 1) } else { 0 }
    Write-Output "$($_.DeviceID) 总计: ${total}GB 已用: ${used}GB(${pct}%) 可用: ${free}GB"
}

Write-Output "--- 分页文件 ---"
Get-CimInstance Win32_PageFileUsage | ForEach-Object {
    Write-Output "$($_.Name) 当前: $($_.CurrentUsage)MB / 分配: $($_.AllocatedBaseSize)MB"
}

Section "CPU & 负载"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Write-Output "CPU: $($cpu.Name)"
Write-Output "核心/线程: $($cpu.NumberOfCores) / $($cpu.NumberOfLogicalProcessors)"
Write-Output "当前负载: $($cpu.LoadPercentage)%"

# ============================================================
# 软件包
# ============================================================
Section "已安装程序 (按大小 TOP30)"

$apps = @()
$regPaths = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)
foreach ($path in $regPaths) {
    $apps += Get-ItemProperty $path 2>$null | Where-Object { $_.DisplayName } |
        Select-Object DisplayName, DisplayVersion, EstimatedSize, Publisher, InstallDate
}
$apps | Sort-Object EstimatedSize -Descending | Select-Object -First 30 |
    Format-Table @{L="名称";E={$_.DisplayName};W=45}, @{L="版本";E={$_.DisplayVersion};W=15},
        @{L="大小(KB)";E={$_.EstimatedSize};W=12}, @{L="发布者";E={$_.Publisher};W=25} -AutoSize |
    Out-String | Write-Output

# Winget
Section "Winget 包"
if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget list 2>$null | Select-Object -First 40
} else {
    Write-Output "winget 不可用"
}

# Chocolatey
Section "Chocolatey 包"
if (Get-Command choco -ErrorAction SilentlyContinue) {
    choco list 2>$null
} else {
    Write-Output "Chocolatey 未安装"
}

# Scoop
Section "Scoop 包"
if (Get-Command scoop -ErrorAction SilentlyContinue) {
    scoop list 2>$null
} else {
    Write-Output "Scoop 未安装"
}

# ============================================================
# 开发环境
# ============================================================
Section "Node.js 环境"
if (Get-Command node -ErrorAction SilentlyContinue) {
    Write-Output "node: $(node -v)"
}
# nvm-windows
if (Get-Command nvm -ErrorAction SilentlyContinue) {
    Write-Output "--- nvm 已安装版本 ---"
    nvm list 2>$null
}
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Write-Output "--- 全局 npm 包 ---"
    npm ls -g --depth=0 2>$null
}
if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    Write-Output "--- 全局 pnpm 包 ---"
    pnpm ls -g 2>$null
    Write-Output "--- pnpm store ---"
    pnpm store path 2>$null
    $pnpmStore = pnpm store path 2>$null
    if ($pnpmStore -and (Test-Path $pnpmStore)) {
        $storeSize = (Get-ChildItem $pnpmStore -Recurse -Force | Measure-Object -Property Length -Sum).Sum
        Write-Output "pnpm store 大小: $([math]::Round($storeSize / 1GB, 2)) GB"
    }
}

Section "Python 环境"
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Output "Python: $(python --version 2>&1)"
    Write-Output "--- pip 全局包 ---"
    pip list --format=columns 2>$null | Select-Object -First 30
}
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Output "--- conda 环境 ---"
    conda env list 2>$null
}

Section "Java 环境"
if (Get-Command java -ErrorAction SilentlyContinue) {
    Write-Output "Java: $(java -version 2>&1 | Select-Object -First 1)"
    Write-Output "JAVA_HOME: $env:JAVA_HOME"
}
# SDKMAN (WSL/Git Bash 里可能有)
if (Test-Path "$env:USERPROFILE\.sdkman\candidates\java") {
    Write-Output "--- SDKMAN 已安装 Java 版本 ---"
    Get-ChildItem "$env:USERPROFILE\.sdkman\candidates\java" -Directory | Select-Object Name
}

Section "Go 环境"
if (Get-Command go -ErrorAction SilentlyContinue) {
    Write-Output "Go: $(go version 2>&1)"
    Write-Output "GOPATH: $env:GOPATH"
    $gopath = go env GOPATH 2>$null
    if ($gopath -and (Test-Path $gopath)) {
        $size = (Get-ChildItem $gopath -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        Write-Output "GOPATH 大小: $([math]::Round($size / 1GB, 2)) GB"
    }
}

# pyenv-win
if (Test-Path "$env:USERPROFILE\.pyenv\pyenv-win\versions") {
    Section "pyenv-win 环境"
    Write-Output "--- pyenv 已安装版本 ---"
    Get-ChildItem "$env:USERPROFILE\.pyenv\pyenv-win\versions" -Directory | Select-Object Name
    $size = (Get-ChildItem "$env:USERPROFILE\.pyenv" -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    Write-Output "pyenv 占用: $([math]::Round($size / 1MB, 1)) MB"
}

# ============================================================
# 容器
# ============================================================
Section "容器"
$ctr = $null
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -eq 0) { $ctr = "docker" }
}
if (-not $ctr -and (Get-Command podman -ErrorAction SilentlyContinue)) { $ctr = "podman" }

if ($ctr) {
    Write-Output "容器运行时: $ctr"
    Write-Output "--- 容器列表 ---"
    & $ctr ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>$null
    if ($LASTEXITCODE -ne 0) { & $ctr ps -a }
    Write-Output "--- 空间占用 ---"
    & $ctr system df 2>$null
    Write-Output "--- 镜像 ---"
    & $ctr images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' 2>$null
    if ($LASTEXITCODE -ne 0) { & $ctr images }
    Write-Output "--- 悬空镜像 ---"
    $dangling = (& $ctr images -f "dangling=true" -q 2>$null | Measure-Object).Count
    Write-Output "悬空镜像数: $dangling"
    Write-Output "--- 同镜像重复容器 ---"
    $images = & $ctr ps -a --format '{{.Image}}' 2>$null
    if ($images) {
        $dupes = $images | Group-Object | Where-Object { $_.Count -gt 1 }
        if ($dupes) { $dupes | ForEach-Object { Write-Output "$($_.Name): $($_.Count) 个容器" } }
        else { Write-Output "无重复" }
    }
    Write-Output "--- 未使用 Volume ---"
    & $ctr volume ls -f dangling=true 2>$null
} else {
    Write-Output "无容器运行时"
}

# WSL
Section "WSL"
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    Write-Output "--- 已安装发行版 ---"
    wsl --list --verbose 2>$null
    Write-Output "--- WSL 虚拟磁盘大小 ---"
    Get-ChildItem "$env:LOCALAPPDATA\Packages\*" -Filter "ext4.vhdx" -Recurse -Force 2>$null |
        ForEach-Object { Write-Output "$($_.FullName): $([math]::Round($_.Length / 1GB, 2)) GB" }
    if (-not (Get-ChildItem "$env:LOCALAPPDATA\Packages\*" -Filter "ext4.vhdx" -Recurse -Force 2>$null)) {
        Write-Output "未找到 WSL 虚拟磁盘"
    }
} else {
    Write-Output "WSL 未安装"
}

# ============================================================
# 进程
# ============================================================
Section "进程 - 内存 TOP30"
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 30 |
    Format-Table @{L="PID";E={$_.Id};W=7}, @{L="名称";E={$_.ProcessName};W=30},
        @{L="内存(MB)";E={[math]::Round($_.WorkingSet64/1MB,1)};W=10},
        @{L="CPU(s)";E={[math]::Round($_.CPU,1)};W=10},
        @{L="启动时间";E={$_.StartTime};W=22} -AutoSize |
    Out-String | Write-Output

Section "进程 - CPU TOP15"
Get-Process | Where-Object { $_.CPU } | Sort-Object CPU -Descending | Select-Object -First 15 |
    Format-Table @{L="PID";E={$_.Id};W=7}, @{L="名称";E={$_.ProcessName};W=30},
        @{L="CPU(s)";E={[math]::Round($_.CPU,1)};W=10},
        @{L="内存(MB)";E={[math]::Round($_.WorkingSet64/1MB,1)};W=10} -AutoSize |
    Out-String | Write-Output

Section "重复/遗留进程检测"
$patterns = @("mcp-server", "mcp_server", "language-server", "code-server", "remote-dev-server")
foreach ($p in $patterns) {
    $hits = Get-Process | Where-Object { $_.ProcessName -match $p -or ($_.Path -and $_.Path -match $p) }
    if ($hits) {
        Write-Output "[$p]:"
        $hits | Format-Table Id, ProcessName, StartTime -AutoSize | Out-String | Write-Output
    }
}
Write-Output "--- 重复应用进程 (同名多实例) ---"
Get-Process | Group-Object ProcessName |
    Where-Object { $_.Count -gt 1 -and $_.Name -match '(java|python|node|ruby|go|dotnet)' } |
    ForEach-Object {
        Write-Output "[$($_.Name)] x $($_.Count) 实例:"
        $_.Group | Format-Table Id, @{L="内存(MB)";E={[math]::Round($_.WorkingSet64/1MB,1)}}, StartTime -AutoSize |
            Out-String | Write-Output
    }

# ============================================================
# 端口
# ============================================================
Section "端口监听"
Get-NetTCPConnection -State Listen 2>$null |
    Sort-Object LocalPort |
    Select-Object -First 50 LocalAddress, LocalPort, OwningProcess,
        @{L="进程名";E={(Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName}} |
    Format-Table -AutoSize | Out-String | Write-Output

Section "对外暴露高危端口"
$highRiskPorts = @(3306, 5432, 27017, 6379, 11211, 9000, 9090, 9200, 9300, 8080, 8443, 15672)
$exposed = Get-NetTCPConnection -State Listen 2>$null |
    Where-Object { $_.LocalAddress -eq "0.0.0.0" -or $_.LocalAddress -eq "::" } |
    Where-Object { $_.LocalPort -in $highRiskPorts }
if ($exposed) {
    $exposed | Select-Object LocalAddress, LocalPort, OwningProcess,
        @{L="进程名";E={(Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName}} |
        Format-Table -AutoSize | Out-String | Write-Output
} else {
    Write-Output "无高危端口对外暴露"
}

# ============================================================
# 服务
# ============================================================
Section "运行中的服务"
Get-Service | Where-Object { $_.Status -eq 'Running' } |
    Sort-Object DisplayName |
    Format-Table @{L="名称";E={$_.Name};W=30}, @{L="显示名称";E={$_.DisplayName};W=50},
        @{L="启动类型";E={$_.StartType};W=12} -AutoSize |
    Out-String | Write-Output

# ============================================================
# 大目录 & 缓存
# ============================================================
Section "大目录 (一层扫描)"
$dirs = @(
    "$env:ProgramFiles",
    "${env:ProgramFiles(x86)}",
    "$env:ProgramData",
    "$env:LOCALAPPDATA",
    "$env:APPDATA"
)
foreach ($d in $dirs) {
    if (Test-Path $d) {
        $size = (Get-ChildItem $d -Directory -Force -ErrorAction SilentlyContinue |
            ForEach-Object {
                $s = 0
                if (Get-Command robocopy -ErrorAction SilentlyContinue) {
                    $output = robocopy $_.FullName NULL /L /S /NJH /NJS /NDL /BYTES /R:0 /W:0 2>$null
                    $bytesLine = $output | Select-String "Bytes\s*:" | Select-Object -First 1
                    if ($bytesLine -match "Bytes\s*:\s*(\d+)") { $s = [long]$Matches[1] }
                } else {
                    $s = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue |
                        Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
                    if ($null -eq $s) { $s = 0 }
                }
                [PSCustomObject]@{ Path = $_.FullName; SizeMB = [math]::Round($s / 1MB, 1) }
            } | Sort-Object SizeMB -Descending | Select-Object -First 5)
        if ($size) {
            Write-Output "--- $d TOP5 子目录 ---"
            $size | Format-Table Path, SizeMB -AutoSize | Out-String | Write-Output
        }
    }
}

Section "缓存与临时文件"
$cacheDirs = @(
    @{ Name = "npm cache"; Path = "$env:APPDATA\npm-cache" },
    @{ Name = "pip cache"; Path = "$env:LOCALAPPDATA\pip\Cache" },
    @{ Name = "gradle cache"; Path = "$env:USERPROFILE\.gradle" },
    @{ Name = "maven cache"; Path = "$env:USERPROFILE\.m2" },
    @{ Name = "nuget cache"; Path = "$env:USERPROFILE\.nuget\packages" },
    @{ Name = "TEMP"; Path = $env:TEMP },
    @{ Name = "Windows Temp"; Path = "$env:SystemRoot\Temp" }
)
foreach ($c in $cacheDirs) {
    if (Test-Path $c.Path) {
        $size = (Get-ChildItem $c.Path -Recurse -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Output "$($c.Name): ${sizeMB} MB  ($($c.Path))"
    }
}

# ============================================================
# 计划任务
# ============================================================
Section "计划任务 (非 Microsoft)"
Get-ScheduledTask 2>$null |
    Where-Object { $_.TaskPath -notmatch '\\Microsoft\\' -and $_.State -ne 'Disabled' } |
    Select-Object -First 30 TaskName, TaskPath, State |
    Format-Table -AutoSize | Out-String | Write-Output

# ============================================================
# 启动项
# ============================================================
Section "启动项"
Get-CimInstance Win32_StartupCommand 2>$null |
    Select-Object Name, Command, Location |
    Format-Table -AutoSize | Out-String | Write-Output

# ============================================================
# Windows Update
# ============================================================
Section "Windows Update 历史 (最近10条)"
try {
    $session = New-Object -ComObject Microsoft.Update.Session
    $searcher = $session.CreateUpdateSearcher()
    $history = $searcher.QueryHistory(0, 10)
    $history | Select-Object Date, Title, @{L="结果";E={
        switch ($_.ResultCode) { 2 {"成功"} 3 {"成功(需重启)"} 4 {"失败"} 5 {"中止"} default {"未知"} }
    }} | Format-Table -AutoSize | Out-String | Write-Output
} catch {
    Write-Output "无法获取更新历史"
}

# ============================================================
# 云环境检测
# ============================================================
Section "云环境检测"
$cloud = "未识别"
# Azure
if (Get-Service WindowsAzureGuestAgent -ErrorAction SilentlyContinue) { $cloud = "Azure" }
# AWS
elseif (Test-Path "C:\ProgramData\Amazon" -ErrorAction SilentlyContinue) { $cloud = "AWS" }
# GCP
elseif (Get-Service GCEAgent -ErrorAction SilentlyContinue) { $cloud = "GCP" }
# 腾讯云
elseif (Test-Path "C:\Program Files\QCloud" -ErrorAction SilentlyContinue) { $cloud = "腾讯云" }
# 阿里云
elseif (Test-Path "C:\ProgramData\aliyun" -ErrorAction SilentlyContinue) { $cloud = "阿里云" }
Write-Output "检测到云环境: $cloud"

if ($cloud -ne "未识别") {
    Write-Output "--- 云厂商相关服务 ---"
    switch ($cloud) {
        "Azure"  { Get-Service | Where-Object { $_.Name -match "WindowsAzure|OMS|waagent" } | Format-Table Name, Status, StartType -AutoSize | Out-String | Write-Output }
        "AWS"    { Get-Service | Where-Object { $_.Name -match "Amazon|aws|SSM" } | Format-Table Name, Status, StartType -AutoSize | Out-String | Write-Output }
        "GCP"    { Get-Service | Where-Object { $_.Name -match "GCE|Google" } | Format-Table Name, Status, StartType -AutoSize | Out-String | Write-Output }
        "腾讯云" { Get-Service | Where-Object { $_.Name -match "QCloud|tat_agent|barad|sgagent" } | Format-Table Name, Status, StartType -AutoSize | Out-String | Write-Output }
        "阿里云" { Get-Service | Where-Object { $_.Name -match "Alibaba|AliYun|aegis|cloudmonitor" } | Format-Table Name, Status, StartType -AutoSize | Out-String | Write-Output }
    }
}

# ============================================================
# 事件日志空间
# ============================================================
Section "事件日志空间"
Get-WinEvent -ListLog * -ErrorAction SilentlyContinue |
    Where-Object { $_.FileSize -gt 0 } |
    Sort-Object FileSize -Descending |
    Select-Object -First 10 @{L="日志名";E={$_.LogName}},
        @{L="大小(MB)";E={[math]::Round($_.FileSize/1MB,1)}},
        @{L="最大(MB)";E={[math]::Round($_.MaximumSizeInBytes/1MB,1)}},
        @{L="条目数";E={$_.RecordCount}} |
    Format-Table -AutoSize | Out-String | Write-Output

# ============================================================
# 防火墙
# ============================================================
Section "防火墙状态"
Get-NetFirewallProfile 2>$null |
    Select-Object Name, Enabled, DefaultInboundAction, DefaultOutboundAction |
    Format-Table -AutoSize | Out-String | Write-Output

# ============================================================
Section "采集完成"
Write-Output "时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "系统: $($os.Caption) $($os.Version) ($($os.OSArchitecture))"
Write-Output "云环境: $cloud"
