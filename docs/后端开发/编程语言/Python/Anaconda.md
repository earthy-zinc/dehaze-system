---
order: 2
---

# Anaconda

## 虚拟环境的操作

### 创建虚拟环境

* 指定环境名称：`conda  create  --name  env_name`
* 创建指定python版本：`conda  create  --name  env_name python=3.5`
* 创建指定python版本下包含某些包：`conda  create  --name  env_name python=3.5 numpy scipy`

### 激活/使用/进入/退出某个虚拟环境

* `conda activate  env_name`
* `conda deactivate `

### 复制某个虚拟环境
`conda  create  --name  new_env_name  --clone  old_env_name`

### 删除某个环境
`conda  remove  --name  env_name  --all`

### 查看当前所有环境
* `conda  info  --envs`
* `conda  env  list`

### 查看当前虚拟环境下的所有安装包

* `conda  list ` (需进入该虚拟环境)
* conda  list  -n  env_name

### 安装或卸载包(进入虚拟环境之后）
* conda  install  xxx
* conda  install  xxx=版本号  # 指定版本号
* conda  install  xxx -i 源名称或链接 # 指定下载源
* conda  uninstall  xxx

### 分享虚拟环境

conda env export > environment.yml  # 导出当前虚拟环境

conda env create -f environment.yml  # 创建保存的虚拟环境

### 导出虚拟环境中所安装的包

* conda list -e > requirements.txt  # 导出
* conda install --yes --file requirements.txt  # 安装

## 虚拟环境镜像源

conda当前的源设置在$HOME/.condarc中，可通过文本查看器查看或者使用命令``conda config --show-sources`查看。

* conda config --show-sources #查看当前使用源
* conda config --remove channels 源名称或链接 #删除指定源
* conda config --add channels 源名称或链接 #添加指定源

### 国内conda源

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

```



## 升级和卸载

升级Anaconda需先升级conda

* `conda  update  conda`
* `conda  update  anaconda`
* `rm  -rf  anaconda`

