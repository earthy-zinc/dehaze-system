// 服务器模式系统依赖：检测缺失并按发行版自动安装，便于在新机器上直接复现同一套环境。
// 只处理 Linux（服务器无桌面）；工作站模式只需 Chrome 与三个 CLI，不涉及这些系统包。
import { IS_LINUX, IS_WIN } from './config.mjs';
import { findInPath, runCommand } from './tools.mjs';

// command 用于探测是否已装；dnf / apt / pacman 给出各包管理器下的包名。
// 补丁版 noVNC 由 install/setup 自行下载，故不在此列（NOVNC_WEB 想用发行版包时另装 novnc）。
const SERVER_DEPENDENCIES = [
  {
    command: 'Xvfb',
    label: 'Xvfb（虚拟显示）',
    dnf: ['xorg-x11-server-Xvfb'],
    apt: ['xvfb'],
    pacman: ['xorg-server-xvfb'],
  },
  { command: 'x11vnc', label: 'x11vnc（VNC 服务端）', dnf: ['x11vnc'], apt: ['x11vnc'], pacman: ['x11vnc'] },
  {
    command: 'websockify',
    label: 'websockify（noVNC 的 ws 网关）',
    dnf: ['python3-websockify'],
    apt: ['websockify'],
    pacman: ['python-websockify'],
  },
  {
    command: 'dbus-daemon',
    label: 'dbus（输入法会话总线）',
    dnf: ['dbus-tools'],
    apt: ['dbus'],
    pacman: ['dbus'],
  },
  {
    command: 'ibus-daemon',
    label: 'ibus + libpinyin（中文输入）',
    dnf: ['ibus', 'ibus-libpinyin', 'ibus-gtk3'],
    apt: ['ibus', 'ibus-libpinyin', 'ibus-gtk3'],
    pacman: ['ibus', 'ibus-libpinyin'],
  },
];

const PACKAGE_MANAGERS = [
  { name: 'dnf', command: 'dnf', args: ['install', '-y'] },
  { name: 'yum', command: 'yum', args: ['install', '-y'] },
  { name: 'apt', command: 'apt-get', args: ['install', '-y'] },
  { name: 'pacman', command: 'pacman', args: ['-S', '--noconfirm'] },
];

export function missingServerDependencies() {
  return SERVER_DEPENDENCIES.filter((dependency) => findInPath(dependency.command) === null);
}

function detectPackageManager() {
  for (const manager of PACKAGE_MANAGERS) {
    if (findInPath(manager.command) !== null) {
      return manager;
    }
  }
  return null;
}

// 非 root 时经 sudo 提权；成功后命令自身的退出码即结果
function runPrivileged(manager, args) {
  const fullArgs = [...manager.args, ...args];
  if (typeof process.getuid === 'function' && process.getuid() === 0) {
    return runCommand(manager.command, fullArgs, { stdio: 'inherit' });
  }
  return runCommand('sudo', [manager.command, ...fullArgs], { stdio: 'inherit' });
}

/**
 * 安装服务器模式缺失的系统依赖（幂等：已装则跳过）
 * @returns {boolean} 是否成功（无缺失或安装成功）
 */
export function installServerDependencies() {
  if (IS_WIN || !IS_LINUX) {
    console.log('系统依赖自动安装仅支持 Linux；工作站模式只需 Chrome 与三个 CLI');
    return true;
  }

  const missing = missingServerDependencies();
  if (missing.length === 0) {
    console.log('服务器系统依赖已齐备');
    return true;
  }

  const manager = detectPackageManager();
  if (manager === null) {
    console.error(`未识别包管理器，请手动安装：${missing.map((item) => item.label).join('、')}`);
    return false;
  }

  const packages = [];
  for (const dependency of missing) {
    const names = dependency[manager.name];
    if (names === undefined) {
      console.error(`⚠️ ${dependency.label} 在 ${manager.name} 下无包名映射，请手动安装`);
      continue;
    }
    packages.push(...names);
  }
  if (packages.length === 0) {
    return false;
  }

  // apt 需先刷新索引，否则新机器上会因缺缓存而找不到包
  if (manager.name === 'apt') {
    console.log('apt-get update…');
    runPrivileged({ ...manager, args: ['update'] }, []);
  }

  console.log(`经 ${manager.name} 安装：${packages.join(' ')}`);
  const result = runPrivileged(manager, packages);
  if (result.status !== 0) {
    console.error(`安装失败（退出码 ${result.status}）`);
    return false;
  }

  const stillMissing = missingServerDependencies();
  if (stillMissing.length > 0) {
    console.error(`安装后仍缺：${stillMissing.map((item) => item.command).join('、')}`);
    return false;
  }
  console.log('服务器系统依赖安装完成');
  return true;
}
