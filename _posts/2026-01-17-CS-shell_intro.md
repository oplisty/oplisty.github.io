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

