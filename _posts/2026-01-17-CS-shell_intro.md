---
title: "Shell的介绍"
date: 2026-01-16 
categories: [Computer science]
tags: [Computer science]
---

## shell 是什么？

如今的计算机有着多种多样的交互接口让我们可以进行指令的输入，从炫酷的图像用户界面（GUI），语音输入甚至是 AR/VR 都已经无处不在。 这些交互接口可以覆盖 80% 的使用场景，但是它们也从根本上限制了您的操作方式——你不能点击一个不存在的按钮或者是用语音输入一个还没有被录入的指令。 为了充分利用计算机的能力，我们不得不回到最根本的方式，使用文字接口：Shell

几乎所有您能够接触到的平台都支持某种形式的 shell，有些甚至还提供了多种 shell 供您选择。虽然它们之间有些细节上的差异，但是其核心功能都是一样的：它允许你执行程序，输入并获取某种半结构化的输出。

本节课我们会使用 Bourne Again SHell, 简称 “bash” 。 这是被最广泛使用的一种 shell，它的语法和其他的 shell 都是类似的。打开 shell *提示符*（您输入指令的地方），您首先需要打开 *终端* 。您的设备通常都已经内置了终端，或者您也可以安装一个，非常简单。

```shell
missing:~$ 
```

1. 主机名是 `missing` 
2. 当前的工作目录 `~` (表示 “home”)
3.  `$` 符号表示您现在的身份不是 root 用户

### 命令执行与环境变量

当你在 shell 中执行命令时，您实际上是在执行一段 shell 可以解释执行的简短代码。如果你要求 shell 执行某个不是 shell 内核所了解到指令的时候，那么它会去查询*环境变量* `$PATH`. 

当我们执行 `echo` 命令时，shell 了解到需要执行 `echo` 这个程序，随后它便会在 `$PATH` 中搜索由 `:` 所分割的一系列目录，基于名字搜索该程序。

```shell
missing:~$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

确定某个程序名代表的是哪个具体的程序，可以使用 `which` 程序。

```shell
missing:~$ which echo
/bin/echo
```

我们也可以绕过 `$PATH`，通过直接指定需要执行的程序的路径来执行该程序

```shell
missing:~$ /bin/echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

### 路径定位

**相对路径**: 相对于当前工作目录的路径，当前工作目录可以使用 `pwd` 命令来获取

**绝对路径**: 路径 `/` 代表的是系统的根目录, 所有的文件夹都包括在这个路径之下，在 Windows 上每个盘都有一个根目录（例如： `C:\`）。 

本课程时使用的是 Linux 系统, 如果某个路径以 `/` 开头，那么它是一个 *绝对路径*，其他的都是 *相对路径* 。

```shell
missing:~$ pwd
/home/missing
missing:~$ cd /home
missing:/home$ pwd
/home
missing:/home$ cd ..
missing:/$ pwd
/
missing:/$ cd ./home
missing:/home$ pwd
/home
missing:/home$ cd missing
missing:~$ pwd
/home/missing
missing:~$ ../../bin/echo hello
hello
```

为了查看指定目录下包含哪些文件，我们使用 `ls` 命令：

```shell
missing:~$ ls
missing:~$ cd ..
missing:/home$ ls
missing
missing:/home$ cd ..
missing:/$ ls
bin
boot
dev
etc
home
...
```

### 参数化

大多数的命令接受标记和选项（带有值的标记），它们以 `-` 开头，并可以改变程序的行为。

**通常，在执行程序时使用 `-h` 或 `--help` 标记可以打印帮助信息来了解有哪些可用的标记或选项。**

```shell 
missing:~$ ls --help 
 -l                         use a long listing format
missing:~$ ls -l /home
drwxr-xr-x 1 missing  users  4096 Jun 15  2019 missing
```

这个参数可以更加详细地列出目录下文件或文件夹的信息

* `drwxr-xr-x`:其中`d` 表明missing 是一个目录,后面9个字符三个三个一组, 分别为**文件所有者权限** ,**用户组权限**,**其他所有人拥有的权限** ,
  * `r`:读
  * `w`:写
  * `x`:可执行权限
  * `-`:无权限

还有几个趁手的命令是您需要掌握的，例如 `mv`（用于重命名或移动文件）、 `cp`（拷贝文件）以及 `mkdir`（新建文件夹）。

如果想要知道关于程序参数、输入输出的信息，或是想要了解它们的工作方式，可以使用 `man` ,它接受一个程序名作为参数，然后将它的文档（用户手册）展现给您。

```shell
missing:~$ man ls
```

**用`q`退出** 

Hash Bang:`#!` :当一个带有 shebang 的文本文件被当作[类 Unix](https://en.wikipedia.org/wiki/Unix-like)操作系统中的可执行文件使用时，[程序加载器](https://en.wikipedia.org/wiki/Loader_(computing))机制会将文件首行的其余部分解析为解释器指令。加载器会执行指定的[解释器](https://en.wikipedia.org/wiki/Interpreter_(computing))程序，并将最初尝试运行脚本时使用的路径作为参数传递给它，以便程序可以使用该文件作为输入数据。

```shell
#!/bin/sh
 curl --head --silent https://missing.csail.mit.edu
```

程序加载器会被指示运行程序*`/bin/sh`*，并将`/bin/sh`作为第一个参数 *传递。*

### IO流

在 shell 中，程序有两个主要的“流”：它们的输入流和输出流。 当程序尝试读取信息时，它们会从输入流中进行读取，当程序打印信息时，它们会将信息输出到输出流中。 通常，一个程序的输入输出流都是您的终端。也就是，您的键盘作为输入，显示器作为输出。 但是，我们也可以重定向这些流！

```shell
missing:~$ echo hello > hello.txt
missing:~$ cat hello.txt
hello
missing:~$ cat < hello.txt
hello
missing:~$ cat < hello.txt > hello2.txt
missing:~$ cat hello2.txt
hello
```

`|` 操作符允许我们将一个程序的输出和另外一个程序的输入连接起来

```shell
missing:~$ ls -l / | tail -n1
drwxr-xr-x 1 root  root  4096 Jun 20  2019 var
missing:~$ curl --head --silent google.com | grep --ignore-case content-length | cut --delimiter=' ' -f2
219
```

### Root User

`root` 是管理员可以创建、读取、更新和删除系统中的任何文件。 通常在我们并不会以根用户的身份直接登录系统，因为这样可能会因为某些错误的操作而破坏系统。我们会在需要的时候使用 `sudo` 命令, 从而以 su（super user 或 root 的简写）的身份执行一些操作。当遇到拒绝访问（permission denied）的错误时，通常是因为此时您必须是根用户才能操作。

**PS: `sudo`必须修饰需要执行的具体操作**

```shell
$ sudo echo 3 > brightness
An error occurred while redirecting file 'brightness'
open: Permission denied
```

这里`echo` 只包含了将输出流倒入的过程,而打开文件的过程是用`shell` 用户打开的所以permission denied，

```shell
 echo 3 | sudo tee brightness
```

## 作业

1. 在 `/tmp` 下新建一个名为 `missing` 的文件夹。
2. 用 `man` 查看程序 `touch` 的使用手册。
3. 用 `touch` 在 `missing` 文件夹中新建一个叫 `semester` 的文件。

```shell
touch semester
```

4. 将以下内容一行一行地写入`semester` 

```
 #!/bin/sh
 curl --head --silent https://missing.csail.mit.edu
```

第一行可能有点棘手， `#` 在 Bash 中表示注释，而 `!` 即使被双引号（`"`）包裹也具有特殊的含义。 单引号（`'`）则不一样，此处利用这一点解决输入问题。更多信息请参考 [Bash quoting 手册](https://www.gnu.org/software/bash/manual/html_node/Quoting.html)

* 尝试执行这个文件。(./semester`）如果程序无法执行，请使用 `ls 命令来获取信息并理解其不能执行的原因。

* 查看 `chmod` 的手册(例如，使用 `man chmod` 命令)

  * 使用 `chmod` 命令改变权限，使 `./semester` 能够成功执行。

    ```shell
    chmod +x semester
    ```

    您的 shell 是如何知晓这个文件需要使用 `sh` 来解析呢？更多信息请参考：[shebang](https://en.wikipedia.org/wiki/Shebang_(Unix))

> [!TIP]
>
> Hash Bang:`#!` :当一个带有 shebang 的文本文件被当作[类 Unix](https://en.wikipedia.org/wiki/Unix-like)操作系统中的可执行文件使用时，[程序加载器](https://en.wikipedia.org/wiki/Loader_(computing))机制会将文件首行的其余部分解析为解释器指令。加载器会执行指定的[解释器](https://en.wikipedia.org/wiki/Interpreter_(computing))程序，并将最初尝试运行脚本时使用的路径作为参数传递给它，以便程序可以使用该文件作为输入数据。
>
> ```shell
> #!/bin/sh
>  curl --head --silent https://missing.csail.mit.edu
> ```
>
> 程序加载器会被指示运行程序*`/bin/sh`*，并将`/bin/sh`作为第一个参数 *传递。*

1. 使用 `|` 和 `>` ，将 `semester` 文件输出的最后更改日期信息，写入主目录下的 `last-modified.txt` 的文件中
