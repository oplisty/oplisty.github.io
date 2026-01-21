---
title: "Docker的基本操作"
date: 2026-01-21
categories: [Computer science]
tags: [Computer science]


---

## Docke容器操作

### 静态运行

**Docker**支持在在启动一个容器同时在容器中执行命令

```shell
docker run [imagename] /bin/echo "Hello world"
```

* `[imagename]`:指定要运行的镜像，Docker 首先从本地主机上查找镜像是否存在，如果不存在，Docker 就会从镜像仓库 Docker Hub 下载公共镜像。
* **/bin/echo "Hello world":** 在启动的容器里执行的命令

我们也可以在后台运行

```shell
docker run -d ubuntu:15.10 /bin/sh -c "while true; do echo hello world; sleep 1; done"
```

**PS**:后台运行的时候不会直接输出运行结果而是输出docker编号

```shell
2b1b7a428627c51
```

我们可以通过`docker logs 2b1b7a428627c51`来查看容器的标准输出

这时如果要进入容器，可以通过以下命令进入：

- **docker attach**：允许你与容器的标准输入（stdin）、输出（stdout）和标准错误（stderr）进行交互。

 ```shell
 $ docker attach 1e560fca3906 
 ```

- **docker exec**：推荐大家使用 docker exec 命令，因为此命令会退出容器终端，但不会导致容器的停止。

  ```shell
  $ docker exec -it 243c32535da7 /bin/bash
  ```

  

### 交互式运行

同时**Docker** 也支持交互式的运行容器

```shell
docker run -i -t ubuntu:15.10 /bin/bash
root@0123ce188bd8:/# #此时我们已进入一个 ubuntu15.10 系统的容器
```

* `-i`: 与标准输入进行交互
* `-t`:在新容器中指定一个终端

### 退出

```shell
root@0123ce188bd8:/#  exit
exit
root@runoob:~# #回到原本主机中
```

退出后容器还在继续运行只是我们退出了但是没有终止,

```shell
runoob@runoob:~$ docker ps
CONTAINER ID        IMAGE                  COMMAND              ...  
5917eac21c36        ubuntu:15.10           "/bin/sh -c 'while t…"    ...
```

输出详情介绍：

**CONTAINER ID:** 容器 ID。

**IMAGE:** 使用的镜像。

**COMMAND:** 启动容器时运行的命令。

**CREATED:** 容器的创建时间。

**STATUS:** 容器状态。

状态有7种：

- created（已创建）
- restarting（重启中）
- running 或 Up（运行中）
- removing（迁移中）
- paused（暂停）
- exited（停止）
- dead（死亡）

我们可以

```shell
docker stop 5917eac21c36    #终止 这时候再用docker ps 就不会出现这个容器了
```

如果想恢复可以

```shell
$ docker start  5917eac21c36
###或者用docker restart 重启
$ docker restart <容器 ID>
```

### 容器的导入导出

```shell
$ docker export 1e560fca3906 > ubuntu.tar
```

导出容器 1e560fca3906 快照到本地文件 ubuntu.tar。

```shell
$ cat docker/ubuntu.tar | docker import - test/ubuntu:v1 #将快照文件 ubuntu.tar 导入到镜像 test/ubuntu:v1
$ docker import http://example.com/exampleimage.tgz example/imagerepo #可以通过指定 URL 或者某个目录来导入.
```

可以使用 docker import 从容器快照文件中再导入为镜像

### 容器的删除

```shell
$ docker rm -f 1e560fca3906
$ docker container prune #清理掉所有处于终止状态的容器。
```



## 镜像image的获取

首先要构建容器就要有image ，那么image从哪来呢？ 1. 自己build ，2.从网上拉

#### 网上拉镜像

在拉镜像之前我们要知道镜像名称 我们可以选择从[Docker Hub]( **https://hub.docker.com/**) 网站搜索镜像也可以使用 docker search 命令来搜索镜像

```shell
runoob@runoob:~$ docker search httpd
```

会输出一个表含有下面的参数

* **NAME:** 镜像仓库源的名称

* **DESCRIPTION:** 镜像的描述

* **OFFICIAL:** 是否 docker 官方发布

* **stars:** 类似 Github 里面的 star，表示点赞、喜欢的意思。

* **AUTOMATED:** 自动构建。

确定好镜像后就可以使用命令 `docker pull` 来下载镜像

```shell
Crunoob@runoob:~$ docker pull ubuntu:13.10
13.10: Pulling from library/ubuntu
6599cadaf950: Pull complete 
23eda618d451: Pull complete 
f0be3084efe9: Pull complete 
52de432f084b: Pull complete 
a3ed95caeb02: Pull complete 
Digest: sha256:15b79a6654811c8d992ebacdfbd5152fcf3d165e374e264076aa435214a947a3
Status: Downloaded newer image for ubuntu:13.10
```

下载完成后，我们可以直接使用这个镜像来运行容器。

```shell
docker run ubuntu:13.10
```

那么我们怎么更新镜像呢基于这个镜像来修改?

```shell
docker run -t -i  ubuntu:13.10
#....一系列操作
exit # 退出容器
docker commit -m="has update" -a="runoob" e218edb10161 runoob/ubuntu:v2 #提交容器
```

​	各个参数说明：

- **-m:** 提交的描述信息
- **-a:** 指定镜像作者
- **e218edb10161：**容器 ID
- **runoob/ubuntu:v2:** 指定要创建的目标镜像名

提交完成后我们可以通过`docker images` 来查看

```shell
runoob@runoob:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
runoob/ubuntu       v2                  70bf1840fd7c        15 seconds ago      158.5 MB
ubuntu              14.04               90d5884b1ee0        5 days ago          188 MB
php                 5.6                 f40e9e0f10c8        9 days ago          444.8 MB
nginx               latest              6f8d099c3adc        12 days ago         182.7 MB
mysql               5.6                 f2e8d6c772c0        3 weeks ago         324.6 MB
httpd               latest              02ef73cf1bc0        3 weeks ago         194.4 MB
ubuntu              15.10               4e3b13c8a266        4 weeks ago         136.3 MB
hello-world         latest              690ed74de00f        6 months ago        960 B
training/webapp     latest              6fae60ef3446        12 months ago       348.8 MB
```

后续就可以直接用新提交的容器来启动一个容器

```shell
runoob@runoob:~$ docker run -t -i runoob/ubuntu:v2 /bin/bash                            
root@1a9fbdeb5da3:/#
```

#### 自己构建镜像

我们使用命令 **docker build** ， 从零开始来创建一个新的镜像。为此，我们需要创建一个 Dockerfile 文件，其中包含一组指令来告诉 Docker 如何构建我们的镜像。

```shell
FROM    centos:6.7
MAINTAINER      Fisher "fisher@sudops.com"

RUN     /bin/echo 'root:123456' |chpasswd
RUN     useradd runoob
RUN     /bin/echo 'runoob:123456' |chpasswd
RUN     /bin/echo -e "LANG=\"en_US.UTF-8\"" >/etc/default/local
EXPOSE  22
EXPOSE  80
CMD     /usr/sbin/sshd -D

```

每一个指令都会在镜像上创建一个新的层，每一个指令的前缀都必须是大写的。

第一条FROM，指定使用哪个镜像源

RUN 指令告诉docker 在镜像内执行命令

然后，我们使用 Dockerfile 文件，通过 docker build 命令来构建一个镜像。

```shell
runoob@runoob:~$ docker build -t runoob/centos:6.7 /path/to/Dockerfile
Sending build context to Docker daemon 17.92 kB
Step 1 : FROM centos:6.7
 ---&gt; d95b5ca17cc3
Step 2 : MAINTAINER Fisher "fisher@sudops.com"
 ---&gt; Using cache
 ---&gt; 0c92299c6f03
Step 3 : RUN /bin/echo 'root:123456' |chpasswd
 ---&gt; Using cache
 ---&gt; 0397ce2fbd0a
Step 4 : RUN useradd runoob
......
```

参数说明：

- **-t** ：指定要创建的目标镜像名
- **.** ：Dockerfile 文件所在目录，可以指定Dockerfile 的绝对路径

创建好镜像后我们就可以可以使用新的镜像来创建容器啦！

#### 删除镜像

指令如下

```shell
$ docker rmi [image-name]
```

## Docker Compose 的编写

```yaml
version: "3.7"

services: #所有的服务都在services下
  app:
    build: 
      context: . #上下文路径
      dockerfile: Dockerfile #指定构建image 的dockerfile 文件名
    command: ["python", "app.py"]        # 可选：覆盖容器启动的默认命令
    entrypoint: /code/entrypoint.sh     #决定了容器启动后跑的可执行文件/脚本及怎么处理你传给容器的参数
    environment: #添加环境变量。您可以使用数组或字典、任何布尔值，布尔值需要用引号引起来，以确保 YML 解析器不会将其转换为 True 或 False。
      APP_ENV: "dev"
      SHOW: "true"
    env_file: # 从文件添加环境变量。可以是单个值或列表的多个值。
      - ./common.env
  		- ./apps/web.env
  		- /opt/secrets.env
    ports: #
      - "8080:8080"                     # 示例页展示了 ports 映射写法
    expose: #暴露端口
      - "8080"                          # 仅容器间可见
    volumes:
      - .:/code
    depends_on: #设置依赖关系,先启动redis才会启动 app
      - redis
    networks:
      backend:
        aliases:
          - myapp
    logging:
      driver: json-file
      options:
        max-size: "200k"
        max-file: "10"
    stop_signal: SIGTERM
    stop_grace_period: 10s

  redis:
    image: "redis:alpine" #指定容器运行的镜像

networks:
  backend: {}

volumes: {}
```



