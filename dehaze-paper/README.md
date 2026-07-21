# 学术论文 (dehaze-paper)

基于高质量码本的双分支多尺度图像去雾方法论文 LaTeX 源码。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/学术论文文档.md)。

## 技术栈

- LaTeX
- TeX Live 2025

## 快速开始

### 1. 安装 TeX Live

#### Windows

1. 下载 [install-tl-windows.exe](https://www.tug.org/texlive/acquire-netinstall.html)，以管理员身份运行
2. 将 `C:\texlive\2025\bin\win32` 添加到系统 PATH

#### Linux

```bash
sudo perl install-tl
export PATH=/usr/local/texlive/2025/bin/x86_64-linux:$PATH
```

#### macOS

```bash
brew install --cask mactex
```

### 2. 验证安装

```bash
tex --version
```

### 3. 编译论文

在项目根目录下依次执行以下命令：

```bash
pdflatex "CMFR-Net.tex"
bibtex "CMFR-Net"
pdflatex "CMFR-Net.tex"
pdflatex "CMFR-Net.tex"
```

编译完成后会在当前目录下生成 `CMFR-Net.pdf`。

### 4. 缺包处理

如编译过程提示缺少宏包，使用 `tlmgr` 安装：

```bash
tlmgr install <package_name>
```

## 注意事项

1. 确保 TeX Live 安装完整，包含所有必要的包
2. 文件编码需使用 UTF-8
