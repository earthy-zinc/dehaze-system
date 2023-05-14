# Jenkins 练习

持续交付流水线，这种流水线定义被写在了一个文本文件jenkinsfile。被提交到项目的源代码中。

这样的话有以下一些好处：

* 自动为所有分支创建流水线构建过程并拉取请求
* 在流水线上进行代码复查和迭代
* 对流水线进行审计和跟踪

## Jenkinsfile

```jenkinsfile
pipeline {
	agent any
	stages {
		stage('build'){
			steps{...
		}
	}
}
```

* pipeline是声明式流水线的一种特定语法，定义包含执行整个流水线的所有内容和指令块
* agent指示Jenkins为整个流水线分配什么执行器和工作区
* stage是描述流水线的一个阶段，一般软件部署的阶段。
* steps描述了在某个阶段上要执行的步骤
* sh 是shell命令
* junit是Java单元测试的流水线阶段

## jenkins和k8s集成

### jenkins端

#### 申请 K8S 凭据

1. 登录 Jenkins，点击右上角「用户」 → 左下角「凭据」，选择全局凭据（Unrestricted），添加凭据，类型选择 X.509 Client Certificate：
   1. Client Key: .kube/config文件中 client-key 对应的 key 文件
   2. Client Certificate: .kube/config文件中 client-certificate 对应的 crt 或是 pem 文件
   3. Server CA Certificate：.kube/config 文件中 certificate-authority 对应的 crt 或是 pem 文件，K8S 的最高权限证书
   4. ID：可不填写，默认会自动生成一串字符串，也可以自行设置
   5. 描述：描述下这个凭据的作用，比如这个可以写 对接 K8S 集群凭据

#### 配置 K8S 集群的对接

登录 Jenkins，点击 系统管理 → 系统配置 → 滑动到页面最下面，点击 a separate configuration page，配置集群cloud：

- Kubernetes 地址：kubernetes服务地址，也就是 apiserver 的地址，一般是master 节点 NodeIP+6443 端口
- Kubernetes 服务证书 key：kube-ca.crt 文件的内容
- 凭据：刚才创建的 certificate 凭据
- Jenkins 地址：Agent 连接 Jenkins Master 的地址

其他都使用默认配置，点击连接测试，连接测试成功，点击 Save 存储。

### **K8S pod template 配置**

Jenkins 的 kubernetes-plugin 在执行构建时会在 kubernetes 集群中自动创建一个 Pod，并在 Pod 内部创建一个名为 jnlp 的容器，该容器会连接 Jenkins 并运行 Agent 程序，形成一个 Jenkins 的 Master 和 Slave 架构，然后 Slave 会执行构建脚本进行构建，但如果构建内容是要创建 Docker Image 就要实现 Docker In Docker 方案（在 Docker 里运行 Docker），如果要在集群内部进行部署操作可以使用 kubectl 执行命令，要解决 kubectl 的安装和权限分配问题。

为了方便配置一个 Pod Templates，在配置 kubernetes 连接内容的下面，这里的模板只是模板（与类一样使用时还要实例化过程），名称和标签列表不要以为是 Pod 的 name 和 label，这里的名称和标签列表只是 Jenkins 查找选择模板时使用的，Jenkins 自动创建 Pod 的 name 是项目名称+随机字母的组合，所以我们填写 jenkins-slave，命名空间填写对应的 namespace。

这边要注意，添加 2 个 container，第一个，Pod 内添加一个容器名称是 jnlp，Docker 镜像填写：jenkins/jnlp-slave:4.3-7，后面的使用默认的即可，然后在添加一个 container，容器名称是 jnlp-kubectl，是这个容器里面有 kubectl 的命令，镜像名称填写 harbor.edu.cn/library/centos-docker-kubectl:v1.0，下面增加了 Host Path Volume：/var/run/docker.sock、/root/.kube/、/etc/kubernetes/pki，这边便是为了 jenkins-slave 下有足够的权限可以执行 docker 及 kubectl 部署到 k8s 集群的权限，因为 jenkins-slave pod 有可能会被调度到任一 worker 节点，所以所有的 worker 节点上都必须有 /root/.kube/、/etc/kubernetes/pki，配置好之后点击保存。

### **Jenkins pipeline 说明**

Pipeline，简单来说，就是一套运行在 Jenkins 上的工作流框架，将原来独立运行于单个或者多个节点的任务连接起来，实现单个任务难以完成的复杂流程编排和可视化的工作。

Jenkins Pipeline 有几个核心概念：

- Node：节点，一个 Node 就是一个 Jenkins 节点，Master 或者 Agent，是执行 Step 的具体运行环境，比如我们之前动态运行的 Jenkins Slave 就是一个 Node 节点
- Stage：阶段，一个 Pipeline 可以划分为若干个 Stage，每个 Stage 代表一组操作，比如：Build、Test、Deploy，Stage 是一个逻辑分组的概念，可以跨多个 Node
- Step：步骤，Step 是最基本的操作单元，可以是打印一句话，也可以是构建一个 Docker 镜像，由各类 Jenkins 插件提供，比如命令：sh 'make'，就相当于我们平时 shell 终端中执行 make 命令一样。

Pipeline的使用：

- Pipeline 脚本是由 Groovy 语言实现的
- Pipeline 支持两种语法：Declarative(声明式)和 Scripted Pipeline(脚本式)语法
- Pipeline 也有两种创建方法：可以直接在 Jenkins 的 Web UI 界面中输入脚本；也可以通过创建一个 Jenkinsfile 脚本文件放入项目源码库中
- 一般我们都推荐在 Jenkins 中直接从源代码控制(SCMD)中直接载入 Jenkinsfile Pipeline 这种方法，但是本次为了更直观的展示，我们在 Web UI 界面中输入脚本

### **完整 pipeline 示例**

部署应用的流程如下：

- 拉取 Github 代码
- maven 打包
- 编写 Dockerfile
- 构建打包 Docker 镜像
- 推送 Docker 镜像到仓库
- 编写 Kubernetes YAML 文件
- 更改 YAML 文件中 Docker 镜像 TAG
- 利用 kubectl 工具部署应用

最终的 Pipeline 脚本如下：

```groovy
pipeline {
    agent none
    stages {
# 这步就是从 GitHub 上拉取代码，注意这边的 GitHub 仓库是 公开的，因为 private 的需要各种权限配置，Jenkins 必须有一个公网 IP 或者是公网域名，但因资源问题，这部分暂时没有办法实现。注意，这边 agent 里面指定运行环境，选择了 master，即是这个步骤在 Jenkins master节点执行。
        stage('Clone Code') {
            agent {
                label 'master'
            }
            steps {
                echo "1.Git Clone Code"
                git url: "https://github.com/0820sdd/prometheus-test-demo.git"
            }
        }
# maven 构建，我们指定了 maven 打包的 agent 是在 Jenkins 所在节点另起一个 docker 容器，容器的 image 为 maven:latest，并且使用 -v 参数把本地的 /root/.m2 挂载到 容器的 /root/.m2  目录下，下面 steps 的步骤即是在这个 maven 容器里面的具体操作：mvn -B clean package -Dmaven.test.skip=true。      
        stage('Maven Build') {
            agent {
                docker {
                    image 'maven:latest'
                    args '-v /root/.m2:/root/.m2'
                }
            }
            steps {
                echo "2.Maven Build Stage"
                sh 'mvn -B clean package -Dmaven.test.skip=true'
            }
        }
# maven 构建成功，下一步就是使用 maven build 生成的 prometheus-test-demo-0.0.1-SNAPSHOT.jar 包进行 docker build，docker build 的具体命令有2条 bash 命令 组成。
        第一步 docker build 使用 -f 指定了 Dockerfile 的文件，使用 --build-arg 参数指定了一些参数，比如上面指定了 jar_name 是 target/prometheus-test-demo-0.0.1-SNAPSHOT.jar，最后使用 -t 参数指定了 docker  build 的 image 的名称及版本号。
        第二步就是 使用 docker tag 命令把上一步 docker build 完成的镜像 打 tag 为 harbor.edu.cn/library/prometheus-test-demo:${BUILD_ID}，这步打 tag 的步骤是为了上传到  harbor 镜像仓库，可以随时使用。
        stage('Image Build') {
            agent {
                label 'master'
            }
            steps {
            echo "3.Image Build Stage"
            sh 'docker build -f Dockerfile --build-arg jar_name=target/prometheus-test-demo-0.0.1-SNAPSHOT.jar -t prometheus-test-demo:${BUILD_ID} . '
            sh 'docker tag  prometheus-test-demo:${BUILD_ID}  harbor.edu.cn/library/prometheus-test-demo:${BUILD_ID}'
            }
        }
# 镜像 build 完成，就可以使用 docker push 命令推送到 harbor.edu.cn 镜像仓库。
        stage('Push') {
            agent {
                label 'master'
            }
            steps {
            echo "4.Push Docker Image Stage"
            sh "docker login --username=admin harbor.edu.cn -p Harbor12345"
            sh "docker push harbor.edu.cn/library/prometheus-test-demo:${BUILD_ID}"
            }
        }
    }
}
# 现在镜像已经打包完成，并推送到了镜像仓库，后面我们所要做的就是拉取 k8s 编排文件，这一步和第一步的 拉取代码实际是一样的，只不过上面的拉取代码是为了 build image，这一步是为了进行部署到 K8S。

注意：这边指定了运行此步骤的节点是在 Jenkins 的 slave 节点下的 jnlp-kubectl container 下，这个 slave 是指在配置 对接 K8S 集群时，在 Pod Template 下指定的 标签列表的名称，必须与这个名称一致，不然 jenkins 执行过程中就会报找不到对应的 label 。还有这边指定了 jnlp-kubectl container ，这是因为 jnlp-kubectl container下有 kubectl 命令，且配置 对接 k8s 集群时，指定了把宿主机的 /root/.kube  /etc/kubernetes/pki 目录分别挂载到 container 的 /root/.kube  /etc/kubernetes/pki目录下，这边就是 jnlp-kubectl container 可以访问 K8S 集群的原因。
node('slave') {
    container('jnlp-kubectl') {
        
        stage('Clone YAML') {
        echo "5. Git Clone YAML To Slave"
        git url: "https://github.com/0820sdd/prometheus-test-demo.git"
        }
# yaml文件拉取完毕，替换其中的变量。        
        stage('YAML') {
        echo "6. Change YAML File Stage"
        sh 'sed -i "s#{VERSION}#${BUILD_ID}#g" ./jenkins/scripts/prometheus-test-demo.yaml'
        }
# 使用 kubectl 命令部署 prometheus-test-demo 应用到 K8S 集群。    
        stage('Deploy') {
        echo "7. Deploy To K8s Stage"
        sh 'kubectl apply -f ./jenkins/scripts/prometheus-test-demo.yaml'
        }
    }
}
```