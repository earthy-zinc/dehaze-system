#Requires -Version 5.1
<#
.SYNOPSIS
    dehaze 开发环境统一管理（Windows）：Scoop(系统级 CLI) + mise(语言运行时) + uv(Python)。

.DESCRIPTION
    幂等、分阶段、可回滚。安装与清理严格分离：
      Install  —— 通用，任何机器（含全新机器）可反复运行，只装不删。
      Cleanup  —— 机器特定，只清理“迁移前的旧装”（读取 legacy 段，全新机器留空即无操作）。
    需要管理员权限的步骤会自动通过 UAC 自提权重启脚本。

.PARAMETER Step
    Install(默认) | Cleanup | All

.PARAMETER Restore
    从最近的 backup\win-* 还原机器/用户 PATH 与环境变量。

.EXAMPLE
    .\setup-toolchain.ps1 -WhatIf          # 预演，不产生任何改动
    .\setup-toolchain.ps1                  # 仅安装（通用）
    .\setup-toolchain.ps1 -Step Cleanup    # 仅清理旧装（机器特定）
    .\setup-toolchain.ps1 -Step All        # 安装 + 清理
    .\setup-toolchain.ps1 -Restore         # 还原 PATH/环境变量

.NOTES
    配置来源：同目录 tooling.config.json（与 Linux 脚本共用）。
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateSet('Install', 'Cleanup', 'All')]
    [string]$Step = 'Install',

    [switch]$Restore,

    [switch]$Elevated
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSCommandPath
$Config = Get-Content (Join-Path $Root 'tooling.config.json') -Raw | ConvertFrom-Json
$BackupRoot = Join-Path $Root ("backup\win-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$LogFile = Join-Path $BackupRoot 'run.log'

function Write-Log {
    param([string]$Message, [ValidateSet('INFO', 'OK', 'WARN', 'ERR', 'STEP', 'WHAT')][string]$Level = 'INFO')
    $color = 'Gray'
    switch ($Level) {
        'OK' { $color = 'Green' }
        'WARN' { $color = 'Yellow' }
        'ERR' { $color = 'Red' }
        'STEP' { $color = 'Cyan' }
        'WHAT' { $color = 'DarkYellow' }
    }
    Write-Host ("[{0}] {1}" -f $Level, $Message) -ForegroundColor $color
}

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-Action {
    param([string]$Description, [scriptblock]$Action)
    if ($WhatIfPreference) {
        Write-Log "WHATIF: $Description" 'WHAT'
        return
    }
    Write-Log $Description 'STEP'
    & $Action
}

function Invoke-Native {
    # 运行外部命令。PowerShell 5.1 会把原生命令写到 stderr 的正常进度当作错误，
    # 在 $ErrorActionPreference='Stop' 下直接中断脚本，故此处临时放宽，并以退出码判断成败。
    param([string]$Description, [scriptblock]$Command)
    if ($WhatIfPreference) {
        Write-Log "WHATIF: $Description" 'WHAT'
        return 0
    }
    Write-Log $Description 'STEP'
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Command 2>&1
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        Write-Log "命令失败 (exit $code)：$Description" 'ERR'
        $output | ForEach-Object { Write-Log "  $_" 'ERR' }
    }
    return $code
}

function Get-PathVariable([string]$Scope) {
    [Environment]::GetEnvironmentVariable('Path', $Scope)
}

function Set-PathVariable([string]$Scope, [string]$Value) {
    [Environment]::SetEnvironmentVariable('Path', $Value, $Scope)
}

function Update-SessionPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath"
}

function Add-PathEntry([string]$Scope, [string]$Entry) {
    $parts = @((Get-PathVariable $Scope) -split ';' | Where-Object { $_ -ne '' })
    if ($parts -notcontains $Entry) {
        Set-PathVariable $Scope (($parts + $Entry) -join ';')
        Write-Log "PATH($Scope) += $Entry" 'OK'
    }
}

function Remove-PathFragments([string]$Scope, [string[]]$Fragments) {
    if (-not $Fragments -or $Fragments.Count -eq 0) { return @() }
    $parts = @((Get-PathVariable $Scope) -split ';' | Where-Object { $_ -ne '' })
    $kept = New-Object System.Collections.Generic.List[string]
    $removed = New-Object System.Collections.Generic.List[string]
    foreach ($p in $parts) {
        $hit = $false
        foreach ($f in $Fragments) {
            if ($p -like "*$f*") { $hit = $true; break }
        }
        if ($hit) { $removed.Add($p) } else { $kept.Add($p) }
    }
    if ($removed.Count -gt 0) {
        Set-PathVariable $Scope ($kept -join ';')
    }
    return $removed
}

function Set-EnvVar([string]$Scope, [string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, $Scope)
}

function Remove-EnvVar([string]$Name) {
    foreach ($scope in @('User', 'Machine')) {
        if ([Environment]::GetEnvironmentVariable($Name, $scope)) {
            [Environment]::SetEnvironmentVariable($Name, $null, $scope)
            Write-Log "移除环境变量 $Name ($scope)" 'OK'
        }
    }
}

function Set-IniFile([string]$Path, [string]$Key, [string]$Value, [string]$Section) {
    # 仅在缺失时写入，避免覆盖用户已有配置
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    # 仅匹配生效行（key=...），避免被模板里的注释行干扰
    if ((Test-Path $Path) -and (Select-String -Path $Path -Pattern ("^\s*" + [regex]::Escape($Key) + "\s*=") -Quiet)) { return }
    $block = if ($Section) { "`n[$Section]`n$Key = $Value" } else { "`n$Key = $Value" }
    Add-Content -Path $Path -Value $block -Encoding UTF8
    Write-Log "写入 $Path ($Key)" 'OK'
}

function Save-Backup {
    Invoke-Action "备份 PATH/环境变量/choco 列表 到 $BackupRoot" {
        New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
        [Environment]::GetEnvironmentVariable('Path', 'Machine') | Set-Content (Join-Path $BackupRoot 'machine-path.txt') -Encoding UTF8
        [Environment]::GetEnvironmentVariable('Path', 'User') | Set-Content (Join-Path $BackupRoot 'user-path.txt') -Encoding UTF8
        foreach ($scope in @('Machine', 'User')) {
            $lines = foreach ($n in @('JAVA_HOME', 'GOROOT', 'GOPATH', 'NVM_HOME', 'NVM_SYMLINK')) {
                "{0}={1}" -f $n, [Environment]::GetEnvironmentVariable($n, $scope)
            }
            $lines | Set-Content (Join-Path $BackupRoot "$($scope.ToLower())-env.txt") -Encoding UTF8
        }
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            choco list 2>$null | Set-Content (Join-Path $BackupRoot 'choco-list.txt') -Encoding UTF8
        }
    }
}

function Restore-Environment {
    $backupDirs = Get-ChildItem (Join-Path $Root 'backup') -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'win-*' } | Sort-Object Name -Descending
    if (-not $backupDirs) { throw "未找到 backup\win-* 备份目录，无法还原。" }
    $src = $backupDirs[0]
    Write-Log "从 $($src.FullName) 还原" 'STEP'
    Invoke-Action '还原机器/用户 PATH 与环境变量' {
        $machinePath = (Get-Content (Join-Path $src.FullName 'machine-path.txt') -Raw).Trim()
        $userPath = (Get-Content (Join-Path $src.FullName 'user-path.txt') -Raw).Trim()
        Set-PathVariable 'Machine' $machinePath
        Set-PathVariable 'User' $userPath
        foreach ($scope in @('Machine', 'User')) {
            foreach ($line in Get-Content (Join-Path $src.FullName "$($scope.ToLower())-env.txt")) {
                $name, $value = $line -split '=', 2
                if ($value) {
                    [Environment]::SetEnvironmentVariable($name, $value, $scope)
                    Write-Log "还原 $name ($scope) = $value" 'OK'
                }
            }
        }
    }
}

function Install-Scoop {
    if (-not (Get-Command scoop -ErrorAction SilentlyContinue)) {
        $scoopCmd = Join-Path $env:USERPROFILE 'scoop\shims\scoop.cmd'
        Invoke-Action '安装 Scoop（走 gitee 镜像，管理员模式）' {
            $installer = Join-Path $env:TEMP 'dehaze-scoop-install.ps1'
            $env:SCOOP_REPO = $Config.mirrors.scoopRepo
            # 安装器默认禁止管理员运行；本会话已提权且为同一用户，显式 -RunAsAdmin 放行
            foreach ($url in @("$($Config.mirrors.scoopRepo)/raw/master/bin/install.ps1", 'https://get.scoop.sh')) {
                Write-Log "使用安装脚本：$url"
                Invoke-RestMethod $url -OutFile $installer
                & $installer -RunAsAdmin
                if (Test-Path $scoopCmd) { break }
            }
            Remove-Item $installer -Force -ErrorAction SilentlyContinue
        }
        Update-SessionPath
        if (-not (Test-Path $scoopCmd)) { throw 'Scoop 安装失败，请检查网络后重试。' }
    }

    Invoke-Native 'Scoop 仓库指向 gitee 镜像' { scoop config SCOOP_REPO $Config.mirrors.scoopRepo } | Out-Null

    # scoop bucket list 走 host 流无法被管道捕获，改用 bucket 目录判断是否已存在
    $bucketRoot = Join-Path $env:USERPROFILE 'scoop\buckets'
    $existing = @()
    if (Test-Path $bucketRoot) {
        $existing = @(Get-ChildItem $bucketRoot -Directory | Select-Object -ExpandProperty Name)
    }
    foreach ($bucket in $Config.windows.buckets) {
        if ($existing -notcontains $bucket) {
            $repo = (Get-Culture).TextInfo.ToTitleCase($bucket)
            Invoke-Native "添加 bucket: $bucket" { scoop bucket add $bucket "$($Config.mirrors.bucketBase)/$repo" } | Out-Null
        }
    }
    if ($Config.windows.useProxyBucket -and ($existing -notcontains 'spc')) {
        Invoke-Native '添加代理 bucket: spc（gh-proxy 加速）' { scoop bucket add spc $Config.mirrors.proxyBucket } | Out-Null
    }

    foreach ($app in $Config.windows.userApps) {
        Invoke-Native "scoop 安装(用户): $app" { scoop install $app } | Out-Null
    }
    Invoke-Native '启用 aria2 多线程下载' { scoop config aria2-enabled true } | Out-Null

    foreach ($app in $Config.windows.globalApps) {
        Invoke-Native "scoop 安装(全局 -g): $app" { scoop install -g $app } | Out-Null
    }

    Invoke-Action '确保全局 shims 在系统 PATH' {
        Add-PathEntry 'Machine' 'C:\ProgramData\scoop\shims'
    }
}

function Configure-Mirrors {
    Invoke-Action '配置 npm/pnpm registry (npmmirror)' {
        Set-IniFile (Join-Path $env:USERPROFILE '.npmrc') 'registry' $Config.mirrors.npmRegistry $null
    }
    Invoke-Action '配置 pip index (tuna)' {
        $pipIni = Join-Path $env:APPDATA 'pip\pip.ini'
        Set-IniFile $pipIni 'index-url' $Config.mirrors.pipIndex 'global'
        Set-IniFile $pipIni 'trusted-host' 'pypi.tuna.tsinghua.edu.cn' 'install'
    }
    Invoke-Action '配置 Maven 镜像 (aliyun，仅当无 settings.xml)' {
        $settings = Join-Path $env:USERPROFILE '.m2\settings.xml'
        if (-not (Test-Path $settings)) {
            $dir = Split-Path -Parent $settings
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
            $xml = @"
<settings>
  <mirrors>
    <mirror>
      <id>aliyun</id>
      <mirrorOf>*</mirrorOf>
      <url>$($Config.mirrors.mavenMirror)</url>
    </mirror>
  </mirrors>
</settings>
"@
            Set-Content -Path $settings -Value $xml -Encoding UTF8
            Write-Log "写入 $settings" 'OK'
        }
    }
}

function Install-Mise {
    # 以 scoop 自己的 shim 判断，避免被 PATH 上其它来源的 mise 干扰而误判“已安装”
    $scoopMise = Join-Path $env:USERPROFILE 'scoop\shims\mise.exe'
    if (-not (Test-Path $scoopMise)) {
        Invoke-Native '安装 mise（scoop）' { scoop install mise } | Out-Null
        Update-SessionPath
    }

    # mise shims 写入用户 PATH —— 让 CMD / Git Bash / IDE 终端等非激活 shell 也能用 node/go/java
    Invoke-Action '确保 mise shims 在用户 PATH（跨 shell 可用）' {
        Add-PathEntry 'User' (Join-Path $env:LOCALAPPDATA 'mise\shims')
    }

    $activateLine = '$env:MISE_PWSH_CHPWD_WARNING = 0' + "`n" + '(& mise activate pwsh) | Out-String | Invoke-Expression'
    # 写激活到 Windows PowerShell 与 PowerShell 7 两个 profile（PS7 存在时才写后者）
    $profilePaths = @((Join-Path $env:USERPROFILE 'Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1'))
    if (Get-Command pwsh -ErrorAction SilentlyContinue) {
        $profilePaths += (Join-Path $env:USERPROFILE 'Documents\PowerShell\Microsoft.PowerShell_profile.ps1')
    }
    foreach ($profilePath in $profilePaths) {
        $hasActivate = (Test-Path $profilePath) -and [bool](Select-String -Path $profilePath -Pattern 'mise activate' -Quiet)
        if ($hasActivate) { continue }
        Invoke-Action "写入 mise 激活到 $profilePath" {
            $profileDir = Split-Path -Parent $profilePath
            if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Force -Path $profileDir | Out-Null }
            if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Force -Path $profilePath | Out-Null }
            Add-Content -Path $profilePath -Value "`n# mise 环境`n$activateLine" -Encoding UTF8
        }
    }

    # Git Bash（MINGW）激活：以 LF 无 BOM 追加，避免污染 .bashrc
    $bashrc = Join-Path $env:USERPROFILE '.bashrc'
    $bashLine = 'command -v mise >/dev/null 2>&1 && eval "$(mise activate bash)"'
    $bashrcText = if (Test-Path $bashrc) { [System.IO.File]::ReadAllText($bashrc) } else { '' }
    if ($bashrcText -notmatch 'mise activate bash') {
        Invoke-Action "写入 mise 激活到 $bashrc" {
            [System.IO.File]::AppendAllText($bashrc, "`n# mise 环境`n$bashLine`n")
        }
    }

    Invoke-Action '设置 mise/go/uv 镜像环境变量' {
        Set-EnvVar 'User' 'MISE_NODE_MIRROR' $Config.mirrors.miseNodeMirror
        Set-EnvVar 'User' 'GOPROXY' $Config.mirrors.goProxy
        Set-EnvVar 'User' 'GOSUMDB' $Config.mirrors.goSumDb
        Set-EnvVar 'User' 'UV_INDEX_URL' $Config.mirrors.pipIndex
    }

    Invoke-Native "mise 安装 node@$($Config.runtimes.node)" { mise use -g "node@$($Config.runtimes.node)" } | Out-Null
    Invoke-Native "mise 安装 go@$($Config.runtimes.go)" { mise use -g "go@$($Config.runtimes.go)" } | Out-Null
    Invoke-Native "mise 安装 java@$($Config.runtimes.java)" { mise use -g "java@$($Config.runtimes.java)" } | Out-Null
}

function Install-Python {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw '未找到 uv。请先确保 scoop 全局应用 uv 安装成功。'
    }
    if ($Config.mirrors.uvPythonMirror) {
        Invoke-Action '设置 uv Python 下载镜像' { Set-EnvVar 'User' 'UV_PYTHON_INSTALL_MIRROR' $Config.mirrors.uvPythonMirror }
    }
    Add-PathEntry 'User' (Join-Path $env:USERPROFILE '.local\bin')
    Invoke-Native "uv 安装 Python $($Config.python.default) 为默认版本" { uv python install $Config.python.default --default } | Out-Null
}

function Install-JsGlobals {
    # JS 全局 CLI 统一走 pnpm（前缀稳定、与 node 版本解耦）；npm 全局前缀固定为安全网，避免 node 升级丢包
    Update-SessionPath
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        Invoke-Native '通过 corepack 启用 pnpm' { corepack enable } | Out-Null
        Update-SessionPath
    }
    $stablePrefix = Join-Path $env:LOCALAPPDATA 'npm-global'
    Invoke-Native '固定 npm 全局前缀到稳定目录' { npm config set prefix $stablePrefix } | Out-Null
    Invoke-Action '确保 npm 固定前缀与 pnpm 全局 bin 在用户 PATH' {
        New-Item -ItemType Directory -Force -Path $stablePrefix | Out-Null
        Add-PathEntry 'User' $stablePrefix
        Add-PathEntry 'User' (Join-Path $env:LOCALAPPDATA 'pnpm')
    }
    foreach ($pkg in $Config.windows.pnpmGlobals) {
        if (-not $pkg) { continue }
        Invoke-Native "pnpm 全局安装: $pkg" { pnpm add -g $pkg } | Out-Null
    }
}

function Invoke-Install {
    Write-Log '=== 安装阶段（通用，可反复运行）===' 'STEP'
    Save-Backup
    Install-Scoop
    Configure-Mirrors
    Install-Mise
    Install-Python
    Install-JsGlobals
    Update-SessionPath
    Write-Log '安装阶段完成。请重开终端（或重启）使 mise/系统 PATH 生效。' 'OK'
}

function Invoke-Cleanup {
    $legacy = $Config.legacy.windows
    Write-Log '=== 清理阶段（机器特定，删除迁移前旧装）===' 'STEP'

    foreach ($pkg in $legacy.choco) {
        $present = choco list 2>$null | Select-String -Pattern ("^" + [regex]::Escape($pkg) + "\s")
        if (-not $present) {
            Write-Log "$pkg 未安装，跳过" 'INFO'
            continue
        }
        Invoke-Native "choco 卸载: $pkg" { choco uninstall $pkg -y --no-progress } | Out-Null
    }

    # 用各自 MSI 卸载器清理旧装（比只删目录更干净，注册表/应用列表一并移除）
    foreach ($code in $legacy.msiProductCodes) {
        if (-not $code) { continue }
        Invoke-Action "MSI 卸载: $code" {
            $proc = Start-Process msiexec.exe -ArgumentList "/x $code /qn /norestart" -Wait -PassThru
            if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 1605 -and $proc.ExitCode -ne 3010) {
                Write-Log "MSI 卸载 $code 返回码 $($proc.ExitCode)" 'WARN'
            }
        }
    }
    foreach ($key in $legacy.registryKeys) {
        if (-not $key) { continue }
        if (Test-Path $key) {
            Invoke-Action "删除注册表键: $key" { Remove-Item -LiteralPath $key -Recurse -Force }
        }
    }

    Invoke-Action '从系统 PATH 移除旧工具条目' {
        $removed = Remove-PathFragments 'Machine' $legacy.machinePathFragments
        foreach ($r in $removed) { Write-Log "移除(机器): $r" 'OK' }
    }
    Invoke-Action '从用户 PATH 移除旧工具条目' {
        $removed = Remove-PathFragments 'User' $legacy.userPathFragments
        foreach ($r in $removed) { Write-Log "移除(用户): $r" 'OK' }
    }
    Invoke-Action '移除冲突的旧环境变量（JAVA_HOME/GOROOT/NVM_* 等）' {
        foreach ($name in $legacy.envVars) { Remove-EnvVar $name }
    }
    foreach ($link in $legacy.links) {
        if (Test-Path $link) {
            # 目录符号链接只能删链接本身，递归删除会误进目标目录
            Invoke-Action "删除旧链接: $link" { [System.IO.Directory]::Delete($link, $false) }
        }
    }
    foreach ($dir in $legacy.dirs) {
        if (Test-Path $dir) {
            Invoke-Action "删除旧安装目录: $dir" {
                try {
                    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop
                }
                catch {
                    # 例如被运行中的 uv/uvx 进程占用；记录后继续，避免中断整个清理
                    Write-Log "删除失败（可能被占用，可稍后/重启后再删）：$dir" 'WARN'
                }
            }
        }
    }

    Write-Log '清理完成。' 'OK'
    Write-Log '请手动检查并重定向以下 IDE 的 SDK 路径：IntelliJ / PyCharm / WebStorm / Android Studio / VS Code。' 'WARN'
}

function Show-Summary {
    Write-Log '=== 结果核对 ===' 'STEP'
    Update-SessionPath
    foreach ($tool in @('scoop', 'mise', 'uv')) {
        $cmd = Get-Command $tool -ErrorAction SilentlyContinue
        if ($cmd) { Write-Log ("{0}: {1}" -f $tool, $cmd.Source) 'OK' } else { Write-Log ("{0}: 未找到" -f $tool) 'WARN' }
    }
    Write-Log "日志：$LogFile"
}

# ---------------------------------------------------------------- 入口
if (-not $WhatIfPreference) {
    New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
    try { Start-Transcript -Path $LogFile -Append | Out-Null } catch { }
}

try {
    if ($Restore) {
        Restore-Environment
        return
    }

    # 需要管理员：安装(scoop -g、系统 PATH) 与 清理均需要；-WhatIf 无改动可免提权
    if (-not (Test-Admin) -and -not $WhatIfPreference -and -not $Elevated) {
        Write-Log '需要管理员权限，正在通过 UAC 提权…' 'WARN'
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Step', $Step, '-Elevated')
        Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait -ArgumentList $argList
        return
    }

    switch ($Step) {
        'Install' { Invoke-Install }
        'Cleanup' { Invoke-Cleanup }
        'All' { Invoke-Install; Invoke-Cleanup }
    }
    Show-Summary
}
finally {
    try { Stop-Transcript | Out-Null } catch { }
}
