# 基于高质量码本的双分支多尺度图像去雾方法

本论文提出了一种创新的图像去雾方法，通过结合高质量码本先验知识和双分支网络结构，有效处理非均匀雾霾场景。主要创新点包括：

1. 使用VQGAN训练高质量码本作为先验知识，补充纹理细节信息
2. 设计金字塔扩张邻域注意力编码器，实现多尺度特征提取  
3. 提出增强解码器，结合像素级和通道级注意力机制
4. 采用双分支网络结构，通过特征融合处理浓雾区域
实验结果表明，该方法在O-HAZE、DENSE-HAZE等多个数据集上均取得了优异性能。

项目包含以下主要文件：

- `A dual branch multi-scale image dehazing method based on high quality codebook.tex`: 主论文文件
- `references.bib`: 参考文献数据库
- 各种实验对比图片：`ablation_experiments.png`, `dehaze_results_1.png`, `dehaze_results_2.png`等

## 环境配置

### 1. 安装TeX Live

#### Windows系统

1. 下载官方安装程序 [install-tl-windows.exe](https://www.tug.org/texlive/acquire-netinstall.html)
2. 以管理员身份运行安装程序
3. 安装完成后，将`C:\texlive\2025\bin\win32`添加到系统PATH环境变量

#### Linux系统

1. 下载安装脚本 [install-tl-unx.tar.gz](https://www.tug.org/texlive/acquire-netinstall.html)
2. 解压并运行安装脚本：

   ```bash
   sudo perl install-tl
   ```

3. 将安装目录下的`bin`路径添加到`.bashrc`或`.zshrc`：

   ```bash
   export PATH=/usr/local/texlive/2025/bin/x86_64-linux:$PATH
   ```

#### macOS系统

1. 使用Homebrew安装：

   ```bash
   brew install --cask mactex
   ```

2. 安装完成后，TeX Live路径已自动配置

### 2. 验证安装

运行以下命令验证安装并检查环境变量配置：

```bash
tex --version
echo $PATH
```

## 编译构建

### 1. 使用命令行编译

#### 1.1 编译论文

在项目根目录下运行以下命令：

```bash
pdflatex "A dual branch multi-scale image dehazing method based on high quality codebook.tex"
```

#### 1.2 处理参考文献

运行以下命令处理参考文献：

```bash
bibtex "A dual branch multi-scale image dehazing method based on high quality codebook"
```

#### 1.3 再次编译

再次运行pdflatex命令两次以确保所有交叉引用正确：

```bash
pdflatex "A dual branch multi-scale image dehazing method based on high quality codebook.tex"
pdflatex "A dual branch multi-scale image dehazing method based on high quality codebook.tex"
```

#### 1.4 查看结果

编译完成后，会在当前目录下生成`A dual branch multi-scale image dehazing method based on high quality codebook.pdf`文件，使用PDF阅读器打开即可查看最终结果。

### 2. 使用VSCode LaTeX Workshop编译

#### 2.1 安装LaTeX Workshop扩展

1. 打开VSCode，进入扩展市场
2. 搜索"LaTeX Workshop"并安装
3. 安装完成后重启VSCode

#### 2.2 配置LaTeX Workshop

1. 打开项目根目录
2. 点击左侧工具栏的"TeX"图标
3. 在设置中确保以下配置：
   - LaTeX: Recipe: latexmk
   - LaTeX: Build: Build LaTeX project
   - LaTeX: Clean: Clean up auxiliary files

#### 2.3 编译项目

1. 打开主TeX文件
2. 按下`Ctrl+Alt+B`（Windows/Linux）或`Cmd+Option+B`（macOS）开始编译
3. 编译完成后，可以在右侧预览PDF文件

## 注意事项

1. 确保TeX Live安装完整，包含所有必要的包
2. 编译过程中可能会提示缺少某些包，可以使用`tlmgr`命令安装：

   ```bash
   tlmgr install <package_name>
   ```

3. 如果遇到编码问题，请确保使用UTF-8编码
