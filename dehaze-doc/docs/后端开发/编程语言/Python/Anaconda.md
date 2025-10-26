---
order: 2
---

# Anaconda

## 虚拟环境操作

### 创建虚拟环境

- 指定环境名称：`conda create --name env_name`
- 创建指定 Python 版本：`conda create --name env_name python=3.5`
- 创建指定 Python 版本下包含某些包：`conda create --name env_name python=3.5 numpy scipy`

### 激活与退出虚拟环境

- 激活环境：`conda activate env_name`
- 退出环境：`conda deactivate`

### 复制虚拟环境

`conda create --name new_env_name --clone old_env_name`

### 删除环境

`conda remove --name env_name --all`

### 查看所有环境

- `conda info --envs`
- `conda env list`

### 查看环境中安装的包

- `conda list` (需进入该虚拟环境)
- `conda list -n env_name`

### 安装或卸载包

进入虚拟环境后执行以下命令：

- 安装包：`conda install xxx`
- 安装指定版本包：`conda install xxx=版本号`
- 指定下载源安装：`conda install xxx -i 源名称或链接`
- 卸载包：`conda uninstall xxx`

### 分享虚拟环境

导出当前虚拟环境：

```bash
conda env export > environment.yml
```

创建保存的虚拟环境：

```bash
conda env create -f environment.yml
```

### 导出环境中安装的包

- 导出：`conda list -e > requirements.txt`
- 安装：`conda install --yes --file requirements.txt`

## 镜像源配置

conda 当前的源设置在 `$HOME/.condarc` 中，可通过文本查看器查看或者使用命令 `conda config --show-sources` 查看。

常用镜像源操作命令：

- 查看当前使用源：`conda config --show-sources`
- 删除指定源：`conda config --remove channels 源名称或链接`
- 添加指定源：`conda config --add channels 源名称或链接`

### 国内镜像源

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
```

## 升级和卸载

升级 Anaconda 需先升级 conda：

- `conda update conda`
- `conda update anaconda`
- `rm -rf anaconda`