---
title: "服务器一些基本操作"
date: 2026-01-18
categories: [Computer science]
tags: [Computer science]
---

用来整理的数据以及相关的应用场景。日志处理通常是一个比较典型的使用场景，因为我们经常需要在日志中查找某些信息，这种情况下通读日志是不现实的。

### `sed`

`sed` 是一个基于文本编辑器 `ed` 构建的 “流编辑器” 。在 `sed` 中，您基本上是利用一些简短的命令来修改文件，而不是直接操作文件的内容（尽管您也可以选择这样做）。相关的命令非常多，但是最常用的是 `s`，即 *替换* 命令，例如我们可以这样写：

```shell
ssh myserver journalctl
 | grep sshd
 | grep "Disconnected from"
 | sed 's/.*Disconnected from //'
```

`s` 命令的语法如下：`s/REGEX/SUBSTITUTION/`, 其中 `REGEX` 部分是我们需要使用的正则表达式，而 `SUBSTITUTION` 是用于替换匹配结果的文本。`sed` 还可以非常方便的做一些事情，例如打印匹配后的内容，一次调用中进行多次替换搜索等。但是这些内容我们并不会在此进行介绍。`sed` 本身是一个非常全能的工具，但是在具体功能上往往能找到更好的工具作为替代品。



### 正则表达式

不同字符所表示的含义，根据正则表达式的实现方式不同，也会有所变化，这一点确实令人沮丧。常见的模式有：

- `.` 除换行符之外的 “任意单个字符”
- `*` 匹配前面字符零次或多次
- `+` 匹配前面字符一次或多次
- `[abc]` 匹配 `a`, `b` 和 `c` 中的任意一个
- `(RX1|RX2)` 任何能够匹配 `RX1` 或 `RX2` 的结果
- `^` 行首
- `$` 行尾

### 结束进程

* `SIGINT` : 当我们输入 `Ctrl-C` 时，shell 会发送一个 `SIGINT` 信号到进程
  * 立刻终止程序且不生成core dump
  * **可以捕获**，比如：
    - Python：`KeyboardInterrupt`
    - C/C++：`signal(SIGINT, handler)`

* `SIGQUIT`:**“更狠”的 Ctrl-C**,通常由 **Ctrl-\** 触发（很少用）,**终止进程**并且**生成 core dump

  * core dump 把进程崩溃时的内存状态保存成文件（用于调试）

  - 可用 `gdb` 分析：

    ```shell
    gdb ./program core
    ```

* `SIGTERM` 则是一个更加通用的、也更加优雅地退出信号。为了发出这个信号我们需要使用 [`kill`](https://www.man7.org/linux/man-pages/man1/kill.1.html) 命令,

  * 标准、推荐的终止信号(终止进程,不生成 core dump)
  * 它的语法是： `kill -TERM <PID>`。

* `SIGTSTP` 信号:键入 `Ctrl-Z` 会让 shell 发送 `SIGTSTP` 信号

  * 它不是让进程退出，而是让进程“暂停”（stop）
  * 可以使用 [`fg`](https://www.man7.org/linux/man-pages/man1/fg.1p.html) 或 [`bg`](https://man7.org/linux/man-pages/man1/bg.1p.html) 命令恢复暂停的工作。它们分别表示在前台继续或在后台继续。

  

[`jobs`](https://man7.org/linux/man-pages/man1/jobs.1p.html) 命令会列出当前终端会话中尚未完成的全部任务。您可以使用 pid 引用这些任务（也可以用 [`pgrep`](https://www.man7.org/linux/man-pages/man1/pgrep.1.html) 找出 pid）。

```shell
$ jobs
[1]+  Stopped   ping 8.8.8.8
$ bg %1  #在后台继续进行(百分号 + 任务编号来选定任务)
$ jobs
[1]+  Running   ping 8.8.8.8 &
```

```shell
$ jobs
[1]+  Stopped   ping 8.8.8.8
$ fg %1  #在前台继续进行
$ jobs
[1]+  Running   ping 8.8.8.8 &
```

训练的时候

```shell
python train.py #前台跑
#发现占着终端但你还要敲别的命令：按 Ctrl+Z 暂停
bg #让它后台继续
fg #想看它的实时输出/或需要交互时
```

命令中的 `&` 后缀可以让命令在直接在后台运行，这使得您可以直接在 shell 中继续做其他操作，不过它此时还是会使用 shell 的标准输出，这一点有时会比较恼人（这种情况可以使用 shell 重定向处理）

```shell
python train.py > train.log & #在后台运行
```

**注意⚠️❗️❗️: 后台的进程仍然是您的终端进程的子进程**，一旦您关闭终端（会发送另外一个信号 `SIGHUP`），这些后台的进程也会终止。为了防止这种情况发生可以使用 [`nohup`](https://www.man7.org/linux/man-pages/man1/kill.1.html)（一个用来忽略 `SIGHUP` 的封装）来运行程序。针对已经运行的程序，可以使用 `disown`

```shell
nohup python train.py > train.log 2>&1 &

```

> 在 Linux/Unix 里，每个进程默认有 3 个输出通道（文件描述符 FD）：
>
> - `0`：标准输入 stdin（默认来自键盘）
> - `1`：标准输出 stdout（默认打印到终端）
> - `2`：标准错误 stderr（默认也打印到终端）
>
> 所以：
>
> - `1` 是“正常输出”
> - `2` 是“报错/警告输出”
>
> - `2>`：对 **FD 2（stderr）** 做重定向
> - `&1`：表示“重定向到 **FD 1（stdout）**”，这里的 `&` 不是后台运行的那个 `&`，而是“把数字当作文件描述符”的标记

```shell
$ python train.py > train.log 2>&1 &
$ jobs
[1]  Running  python train.py > train.log 2>&1 &
$ disown %1 #把它从当前 shell 管理中移除：
#现在你可以直接关终端/断开 SSH，作业一般不会跟着死。
```

### 别名

利用`alias` 可以为长命令设置别名,shell 的别名相当于一个长命令的缩写，shell 会自动将其替换成原本的命令。

```shell
alias alias_name="command_to_alias arg1 arg2"
```

```shell
# 创建常用命令的缩写
alias ll="ls -lh"

# 能够少输入很多
alias gs="git status"
alias gc="git commit"
alias v="vim"

# 手误打错命令也没关系
alias sl=ls

# 重新定义一些命令行的默认行为
alias mv="mv -i"           # -i prompts before overwrite
alias mkdir="mkdir -p"     # -p make parent dirs as needed
alias df="df -h"           # -h prints human readable format

# 别名可以组合使用
alias la="ls -A"
alias lla="la -l"

# 在忽略某个别名
\ls
# 或者禁用别名
unalias la

# 获取别名的定义
alias ll
# 会打印 ll='ls -lh'
```

值得注意的是，在默认情况下 shell 并不会保存别名。为了让别名持续生效，需要将配置放进 shell 的启动文件里，像是 `.bashrc` 或 `.zshrc`

### 配置文件

**点文件**:文件名以 `.` 开头，例如 `~/.vimrc`的文件称为$Dot\ Files$  经常作为配置文件存在(它们默认是隐藏文件，`ls` 并不会显示它们)

shell 的配置也是通过这类文件完成的。在启动时，您的 shell 程序会读取很多文件以加载其配置项。根据 shell 本身的不同，您从登录开始还是以交互的方式完成这一过程可能会有很大的不同

 `bash` 通过编辑 `.bashrc` 或 `.bash_profile` 来进行配置。在文件中可以添加需要在启动时执行的命令，例如上文我们讲到过的别名，或者环境变量。

> 实际上，很多程序都要求您在 shell 的配置文件中包含一行类似 `export PATH="$PATH:/path/to/program/bin"` 的命令，这样才能确保这些程序能够被 shell 找到。

下面列举一些常用的配置

- `bash` - `~/.bashrc`, `~/.bash_profile`
- `git` - `~/.gitconfig`
- `vim` - `~/.vimrc` 和 `~/.vim` 目录
- `ssh` - `~/.ssh/config`
- `tmux` - `~/.tmux.conf`

**配置文件管理：** 用版本控制系统进行管理，然后通过脚本将其 **符号链接** 到需要的地方这么做有如下好处：

- **安装简单**: 如果您登录了一台新的设备，在这台设备上应用您的配置只需要几分钟的时间；
- **可移植性**: 您的工具在任何地方都以相同的配置工作
- **同步**: 在一处更新配置文件，可以同步到其他所有地方
- **变更追踪**: 您可能要在整个程序员生涯中持续维护这些配置文件，而对于长期项目而言，版本历史是非常重要的

**配置文件中需要放些什么**: 作者们在文章中会分享他们的配置。还有一种方法就是直接浏览其他人的配置文件：您可以在这里找到无数的 [dotfiles 仓库](https://github.com/search?o=desc&q=dotfiles&s=stars&type=Repositories) —— 其中最受欢迎的那些可以在 [这里](https://github.com/mathiasbynens/dotfiles) 找到（建议您不要直接复制别人的配置）。[这里](https://dotfiles.github.io/) 也有一些非常有用的资源。

**有时您仅希望特定的配置只在某些设备上生效,可以通过if的shell脚本语言来编写, 或者`include`功能** 

```shell
if [[ "$(uname)" == "Linux" ]]; then {do_something}; fi

# 使用和 shell 相关的配置时先检查当前 shell 类型
if [[ "$SHELL" == "zsh" ]]; then {do_something}; fi

# 您也可以针对特定的设备进行配置
if [[ "$(hostname)" == "myServer" ]]; then {do_something}; fi
```

```shell
[include]
    path = ~/.gitconfig_local
```

