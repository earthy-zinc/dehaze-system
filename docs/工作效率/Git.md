# Git 教程

## 一、工作流程

### 1、.gitignore文件

每个带有git功能的项目都会有一个.gitignore文件，.gitignore文件的作用是告诉git哪些文件不需要添加到版本管理中。常用的过滤规则有：

* /directory/ 过滤掉名为directory的整个文件夹及其内容
* /directory 仅过滤掉项目根目录的directory文件夹及其内容而不包括子文件夹中含有directory文件夹
* directory/ 过滤掉所有名为directory的目录下所有的文件
* *.file_extension 过滤掉所有文件类型为file_extension的文件
* /directory/filename.file_extension 过滤掉directory文件夹中文件名为filename，扩展名为file_extension的文件
* 如果以`!`开头的行，说明gitignore不忽略这些文件夹或者文件，也就是指定这些文件上传到git中
* 以 `#`开头的行为注释，将被 gitignore 忽略

## 二、git本地操作

### 1、创建git项目

有两种方式能够创建一个git管理的项目：

1. 将尚未进行版本控制的本地目录转换为git仓库
2. 从其他服务器上克隆一个已经存在的git仓库到本机上

init 将一个目录初始化为git管理的代码项目

```bash
git init 			# 在你的项目根目录下
```

clone 复制一个项目或者说克隆一个项目

```bash
git clone [url]		# 复制远程仓库中的所有代码到本地的一个目录中
```

### 2、添加到本地缓存

add 在向git本地库提交操作之前，你需要添加文件到缓存中

```shell
git add [filename1] [filename2] ...
git add [directory]
git add .		#添加当前目录下的所有文件到暂存区，会根据gitignore过滤
git add *		#会忽略.gitignore文件，直接将任何文件加入
```

status 查看仓库当前的状态，显示有变更的文件。

diff 比较文件的不同，即缓存区和当前工作区的差异。

```shell
git diff			#显示暂存区和当前工作区的差异
git diff --cached    #显示暂存区和上一次提交的差异
git diff [first_branch] [second_branch] #显示两次提交之间的差异
```





reset 回退版本

rm 将文件从缓存区中删除

### 3、提交到本地库

commit 将对文件做出的所有改动提交到git的本地仓库中

```shell
git commit -m "your comment"
git commit [file1] [file2] ... -m "your comment" #提交指定文件到仓库
git commit -a 								  #无需添加到缓存，直接提交
```

### 4、分支与合并操作

branch 创建分支，但是并不会自动切换到该分支上

```shell
git branch your_branch_name
```

* -b 创建完分支自动切换到分支中，相当于两条命令
* -d 删除该分支
* 无参数 显示当前所有分支列表

checkout 切换到某个分支

```shell
git checkout your_branch_name
```



merge 合并分支，需要先切换到主分支上，然后用命令合并你想合并的那个副分支，那么副分支就会合并到主分支中

```shell
git checkout your_master_branch
git merge your_sub_branch
```





log 显示一个分支中提交的记录中的更改情况

```shell
git log
```

在不传入任何参数的情况下，默认会按时间先后顺序列出所有提交，最近更新的排在最上面，

* -p --patch 显示每次提交所引入的差异
* -n 其中n为数字，限制显示日志条目数量
* --stat 显示每次提交的文件修改统计信息
* --name-only 只显示提交信息后面修改过的文件名称



tag 给历史记录的某个重要的地方打上标签



## 三、git分享操作

fetch 从远程仓库下载分支和数据

pull 从远程仓库更新数据到本地并尝试合并到当前分支

push 将你的代码分支推送到某个远程仓库中

remote 与远程仓库相关的命令，如列出远程仓库名称、添加或者删除一个远程仓库

