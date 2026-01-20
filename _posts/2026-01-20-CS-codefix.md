---
title: "代码调试"
date: 2026-01-20
categories: [Computer science]
tags: [Computer science]


---

# 调试代码

## 打印调试法与日志

“最有效的 debug 工具就是细致的分析，配合恰当位置的打印语句” — Brian Kernighan, *Unix 新手入门*。

**方法1**:在发现问题的地方添加一些打印语句，然后不断重复此过程直到您获取了足够的信息并找到问题的根本原因。

**方法2**(使用日志): 对于一个程序

```shell
oplisty@oplistydeMacBook-Air oplisty.github.io % python3 /Users/oplisty/Downloads/logger.py
Value is 4 - Everything is fine
Value is 8 - Dangerous region
Value is 2 - Everything is fine
Value is 3 - Everything is fine
Maximum value reached
Maximum value reached
Value is 2 - Everything is fine
Value is 3 - Everything is fine
Value is 4 - Everything is fine
Value is 1 - Everything is fine
Value is 5 - System is getting hot
```

可以用日志的形式呈现

```shell
oplisty@oplistydeMacBook-Air oplisty.github.io % python3 /Users/oplisty/Downloads/logger.py log
2026-01-20 19:10:08,753 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:10:09,058 : WARNING : Sample : Value is 5 - System is getting hot
2026-01-20 19:10:09,363 : INFO : Sample : Value is 1 - Everything is fine
2026-01-20 19:10:09,664 : ERROR : Sample : Value is 7 - Dangerous region
2026-01-20 19:10:09,969 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:10:10,273 : ERROR : Sample : Value is 8 - Dangerous region
2026-01-20 19:10:10,578 : ERROR : Sample : Value is 8 - Dangerous region
2026-01-20 19:10:10,880 : ERROR : Sample : Value is 7 - Dangerous region
2026-01-20 19:10:11,185 : INFO : Sample : Value is 4 - Everything is fine
2026-01-20 19:10:11,491 : INFO : Sample : Value is 4 - Everything is fine
2026-01-20 19:10:11,796 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:10:12,101 : INFO : Sample : Value is 1 - Everything is fine
2026-01-20 19:10:12,407 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:10:12,712 : ERROR : Sample : Value is 8 - Dangerous region
2026-01-20 19:10:13,017 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:10:13,322 : INFO : Sample : Value is 4 - Everything is fine
2026-01-20 19:10:13,623 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:10:13,929 : INFO : Sample : Value is 3 - Everything is fine
2026-01-20 19:10:14,231 : INFO : Sample : Value is 4 - Everything is fine
2026-01-20 19:10:14,536 : INFO : Sample : Value is 1 - Everything is fine
2026-01-20 19:10:14,841 : INFO : Sample : Value is 4 - Everything is fine
2026-01-20 19:10:15,146 : ERROR : Sample : Value is 7 - Dangerous
```

可以筛选出ERROR 严重等级以上的条目

```shell
oplisty@oplistydeMacBook-Air oplisty.github.io % python3 /Users/oplisty/Downloads/logger.py log 
ERROR
2026-01-20 19:12:51,481 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:12:51,786 : ERROR : Sample : Value is 7 - Dangerous region
2026-01-20 19:12:52,092 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:12:54,214 : ERROR : Sample : Value is 7 - Dangerous region
2026-01-20 19:12:54,519 : CRITICAL : Sample : Maximum value reached
2026-01-20 19:12:55,129 : ERROR : Sample : Value is 8 - Dangerous region
2026-01-20 19:12:58,167 : ERROR : Sample : Value is 7 - Dangerous region
2026-01-20 19:12:58,771 : CRITICAL : Sample : Maximum value reached
```

也可以用颜色标注

```shell
python logger.py color
```

那怎么设置彩色输出呢

```shell
echo -e "\e[38;2;255;0;0mThis is red\e[0m"
```

`ls` 和 `grep` 这样的程序会使用 [ANSI escape codes](https://en.wikipedia.org/wiki/ANSI_escape_code)，它是一系列的特殊字符，可以使您的 shell 改变输出结果的颜色。例如，执行 `echo -e "\e[38;2;255;0;0mThis is red\e[0m"` 会打印红色的字符串：`This is red` 。只要您的终端支持 [真彩色](https://gist.github.com/XVilka/8346728#terminals--true-color)。如果您的终端不支持真彩色（例如 MacOS 的 Terminal.app），您可以使用支持更加广泛的 16 色，例如：”\e[31; 1mThis is red\e[0m “

写法

```shell
真彩色前景：\e[38;2;R;G;Bm 你要打印的字符 \e[0m

真彩色背景：\e[48;2;R;G;Bm 你要打印的字符 \e[0m

其中\e[0m为复位字符
```

## 系统日志

大多数的程序都会将日志保存在您的系统中的某个地方。对于 UNIX 系统来说，程序的日志通常存放在 `/var/log` 

目前，系统开始使用 **system log**，您所有的日志都会保存在这里。大多数（但不是全部的）Linux 系统都会使用 `systemd`，这是一个系统守护进程，它会控制您系统中的很多东西，例如哪些服务应该启动并运行。`systemd` 会将日志以某种特殊格式存放于 `/var/log/journal`，您可以使用 [`journalctl`](https://man7.org/linux/man-pages/man1/journalctl.1.html) 命令显示这些消息。

对于大多数的 UNIX 系统，您也可以使用 [`dmesg`](https://man7.org/linux/man-pages/man1/dmesg.1.html) 命令来读取内核的日志。

```shell
sudo dmesg
```

如果您希望将日志加入到系统日志中，您可以使用 [`logger`](https://man7.org/linux/man-pages/man1/logger.1.html) 这个 shell 程序。

```shell
logger "Hello Logs"
# On macOS
log show --last 1m | grep Hello
# On Linux
journalctl --since "1m ago" | grep Hello
```

## 调试器

回到调试代码的这个Topic ,因为 My major 是AI 基本上所有的research work 都focus on python 所以我注重讲解 Python 的调试器 [`pdb`](https://docs.python.org/3/library/pdb.html).

首先什么是调试器呢?调试器是一种可以允许我们和正在执行的程序进行交互的程序，它可以做到：

- 当到达某一行时将程序暂停；
- 一次一条指令地逐步执行程序；
- 程序崩溃后查看变量的值；
- 满足特定条件时暂停程序；
- 其他高级功能。

下面对 `pdb` 的使用方法进行简单的介绍：

```shell
python -m pdb train.py
```

- **l**(ist) - 显示当前行周围的 11 行，或接着上次显示的，继续往下显示 11 行；
- **s**(tep) - 执行当前行，并在第一个可能的时机停止（通常指步入函数）；
- **n**(ext) - 继续执行，直到到达当前函数的下一行或函数返回（通常指步过函数）；
- **b**(reak) - 设置断点（基于传入的参数）；
- **p**(rint) - 在当前上下文对表达式求值并打印结果。还有一个 **pp** 命令，它使用 [`pprint`](https://docs.python.org/3/library/pprint.html) 打印；
- **r**(eturn) - 执行到当前函数返回；
- **q**(uit) - 退出调试器。

[`ipdb`](https://pypi.org/project/ipdb/) 是 `pdb` 的增强版，它使用 [`IPython`](https://ipython.org/) 作为 REPL 并开启了 tab 补全、语法高亮、更好的回溯和更好的内省，同时保留了与 `pdb` 模块相同的接口。

在你想停的地方写：

```
import ipdb; ipdb.set_trace()
```

运行程序到这行会进入交互调试界面（提示符一般是 `ipdb>`）。

```python
def f(x):
    y = x * 2
    import ipdb; ipdb.set_trace()
    return y + 1

print(f(10))
```

运行的时候到了第三行会自动打断点

```shell
~/zepeng$ python Agent_Framework.py
> Agent_Framework.py(4)f()
      3     import ipdb; ipdb.set_trace()
----> 4     return y + 1
      5 

```

具体操作如`pdb` 一样

对于更底层的编程语言，可以了解一下 [`gdb`](https://www.gnu.org/software/gdb/)（及其改进版 [`pwndbg`](https://github.com/pwndbg/pwndbg)）和 [`lldb`](https://lldb.llvm.org/)。它们针对类 C 语言的调试进行了优化，也允许探索几乎任何进程并获取其当前的机器状态，例如：寄存器、栈、程序计数器等。

## 专门工具

针对调试的程序是一个二进制的黑盒程序的情况

#### 系统调用调试

当您的程序需要执行一些只有操作系统内核才能完成的操作时，它需要使用 [系统调用](https://en.wikipedia.org/wiki/System_call)。有一些命令可以帮助您追踪您的程序执行的系统调用。在 Linux 中可以使用 [`strace`](https://man7.org/linux/man-pages/man1/strace.1.html) ，在 macOS 和 BSD 中可以使用 [`dtrace`](http://dtrace.org/blogs/about/)（也可以使用使用一个叫做 [`dtruss`](https://www.manpagez.com/man/1/dtruss/) 的封装使其具有和 `strace` (更多信息参考 [这里](https://8thlight.com/blog/colin-jones/2015/11/06/dtrace-even-better-than-strace-for-osx.html))类似的接口

```shell
# On Linux
sudo strace -e lstat ls -l > /dev/null
# On macOS
sudo dtruss -t lstat64_extended ls -l > /dev/null
```

#### 网络数据包定位问题

 [`tcpdump`](https://man7.org/linux/man-pages/man1/tcpdump.1.html) 和 [Wireshark](https://www.wireshark.org/) 这样的网络数据包分析工具可以帮助您获取网络数据包的内容并基于不同的条件进行过滤。

对于 web 开发， Chrome/Firefox 的开发者工具非常方便，功能也很强大：

- 源码 -查看任意站点的 HTML/CSS/JS 源码；
- 实时修改 HTML, CSS, JS 代码 - 修改网站的内容、样式和行为用于测试（由此可见网页截图是不可信的）；
- Javascript shell - 在 JS REPL 中执行命令；
- 网络 - 分析请求的时间线；
- 存储 - 查看 Cookies 和本地应用存储。

##  静态分析

静态分析会将程序的源码作为输入然后基于规则对其进行分析并对代码的正确性进行推理。**静态分析不跑代码只是观察来看代码的错误**

主要有两个工具:

* [`pyflakes`](https://pypi.org/project/pyflakes) 
* [`mypy`](https://mypy-lang.org/) 
* [`shellcheck`](https://www.shellcheck.net/):应用于`shell` 脚本

对于代码

```python
import time

def foo():
    return 42

for foo in range(5):
    print(foo)
bar = 1
bar *= 0.2
time.sleep(60)
print(baz)
```

分别用`pyflakes`和`mypy`进行数据分析可以得到

```
$ pyflakes foobar.py
foobar.py:6: redefinition of unused 'foo' from line 3
foobar.py:11: undefined name 'baz'

$ mypy foobar.py
foobar.py:6: error: Incompatible types in assignment (expression has type "int", variable has type "Callable[[], Any]")
foobar.py:9: error: Incompatible types in assignment (expression has type "float", variable has type "int")
foobar.py:11: error: Name 'baz' is not defined
Found 3 errors in 1 file (checked 1 source file)
```

#### **代码格式化工具（code formatters）**

例如 Python 的 [`black`](https://github.com/psf/black)、Go 语言的 `gofmt`、Rust 的 `rustfmt` 以及 JavaScript、HTML 和 CSS 的 [`prettier`](https://prettier.io/)。这些工具会自动格式化您的代码，使其符合该编程语言通用的风格规范。

**标准化的代码格式不仅有助于他人阅读您的代码，也能让您更轻松地阅读他人（同样经过格式化）的代码, 因为本人有强迫症所以所有代码完成后都会格式化一下** 

```shell
black path/to/script.py #格式化单个文件
black . #格式化整个目录
```

可以在项目的用`pyproject.toml`文件配置

```toml
[tool.black]
line-length = 88
target-version = ["py310"]
exclude = '''
/(
  \.git
| \.venv
| build
| dist
)/
'''
```

# 性能分析

即使你的代码在功能上完全符合预期，但如果它在运行过程中耗尽了所有的 CPU 或内存资源，那也未必合格。算法课程通常会教授大 O 表示法，却很少教你如何找到程序中的热点 (hot spots).

#### 计时

只需打印代码从一处运行到另一处的时间，即可发现问题。下面是一个使用 Python [`time`](https://docs.python.org/3/library/time.html) 模块的例子.

```python
import time, random
n = random.randint(1, 10) * 100

# 获取当前时间 
start = time.time()

# 做些工作
print("Sleeping for {} ms".format(n))
time.sleep(n/1000)

# 比较当前时间和起始时间
print(time.time() - start)

# 输出
# Sleeping for 500 ms
# 0.5713930130004883
```

不过，墙上时间（wall clock time）也可能会有误导性，因为计算机可能同时在运行其他进程，或者在等待某些事件发生。工具通常会区分实际时间、用户时间和系统时间。通常用户时间加系统时间代表了您的进程在 CPU 上实际消耗了多少时间（更详细的解释可以参考 [这篇文章](https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1)）。

- 真实时间 *Real* - 程序从开始到结束流逝的墙上时间，包括其他进程使用的时间以及阻塞（例如等待 I/O 或网络）的时间
- 用户时间 *User* - CPU 执行用户态代码所花费的时间
- 系统时间 *Sys* - CPU 执行内核态代码所花费的时间

例如，试着写一个执行 HTTP 请求的命令，并在命令前加上 [`time`](http://man7.org/linux/man-pages/man1/time.1.html)。网络不好的情况下您可能会看到请求花费了 2 秒多才完成，但是进程仅花费了 15 毫秒的 CPU 用户时间和 12 毫秒的 CPU 内核时间。

#### 性能分析工具

##### CPU

CPU 性能分析工具有两种： 追踪分析器（*tracing*）及采样分析器（*sampling*）

* 追踪分析器:记录程序的每一次函数调用
* 采样分析器:只会周期性的监测（通常为每毫秒）您的程序并记录程序堆栈

它们使用这些记录来生成统计信息，显示程序在哪些事情上花费了最多的时间。如果希望了解更多相关信息，可以参考 [这篇](https://jvns.ca/blog/2017/12/17/how-do-ruby---python-profilers-work-) 介绍性的文章。

在 Python 中

###### `cProfile` 模块来分析每次函数调用所消耗的时间

用法:

```shell
python -m cProfile -s tottime pyfile
```



```shell
$ python -m cProfile -s tottime grep.py 1000 '^(import|\s*def)[^,]*$' *.py

[omitted program output]

 ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   8000    0.266    0.000    0.292    0.000 {built-in method io.open}
   8000    0.153    0.000    0.894    0.000 grep.py:5(grep)
  17000    0.101    0.000    0.101    0.000 {built-in method builtins.print}
   8000    0.100    0.000    0.129    0.000 {method 'readlines' of '_io._IOBase' objects}
  93000    0.097    0.000    0.111    0.000 re.py:286(_compile)
  93000    0.069    0.000    0.069    0.000 {method 'search' of '_sre.SRE_Pattern' objects}
  93000    0.030    0.000    0.141    0.000 re.py:231(compile)
  17000    0.019    0.000    0.029    0.000 codecs.py:318(decode)
      1    0.017    0.017    0.911    0.911 grep.py:3(<module>)

[omitted lines]
```

关于 Python 的 `cProfile` 分析器（以及其他一些类似的分析器），需要注意的是它显示的是每次函数调用的时间。看上去可能快到反直觉，尤其是如果在代码里面使用了第三方的函数库，因为内部函数调用也会被看作函数调用。**所以当繁琐调用的时候你可能得到超过 2500 行的输出结果，即使对其进行排序，我仍然搞不懂时间到底都花在哪了** 

######  [`line_profiler`](https://github.com/pyutils/line_profiler)

它会基于行来显示时间

用法:

```shell
kernprof -l -v pyfile
```

输出如下

```shell
$ kernprof -l -v a.py
Wrote profile results to urls.py.lprof
Timer unit: 1e-06 s

Total time: 0.636188 s
File: a.py
Function: get_urls at line 5

Line #  Hits         Time  Per Hit   % Time  Line Contents
==============================================================
 5                                           @profile
 6                                           def get_urls():
 7         1     613909.0 613909.0     96.5      response = requests.get('https://missing.csail.mit.edu')
 8         1      21559.0  21559.0      3.4      s = BeautifulSoup(response.content, 'lxml')
 9         1          2.0      2.0      0.0      urls = []
10        25        685.0     27.4      0.1      for url in s.find_all('a'):
11        24         33.0      1.4      0.0          urls.append(url['href'])

```

**PS**: 函数前面一定要有`@profile`这个修饰器来告诉行分析器我们想要分析这个函数

##### 内存

像 C 或者 C++ 这样的语言，内存泄漏会导致您的程序在使用完内存后不去释放它。为了应对内存类的 Bug，我们可以使用类似 [Valgrind](https://valgrind.org/) 这样的工具来检查内存泄漏问题。

对于 Python 这类具有垃圾回收机制的语言，内存分析器也是很有用的，因为对于某个对象来说，只要有指针还指向它，那它就不会被回收。

###### [memory-profiler](https://pypi.org/project/memory-profiler/)

```shell
$ python -m memory_profiler example.py
Line #    Mem usage  Increment   Line Contents
==============================================
     3                           @profile
     4      5.97 MB    0.00 MB   def my_func():
     5     13.61 MB    7.64 MB       a = [1] * (10 ** 6)
     6    166.20 MB  152.59 MB       b = [2] * (2 * 10 ** 7)
     7     13.61 MB -152.59 MB       del b
     8     13.61 MB    0.00 MB       return a
```

##### 事件分析

在我们使用 `strace` 调试代码的时候，您可能会希望忽略一些特殊的代码并希望在分析时将其当作黑盒处理。[`perf`](http://man7.org/linux/man-pages/man1/perf.1.html) 命令将 CPU 的区别进行了抽象，它不会报告时间和内存的消耗，而是报告与您的程序相关的系统事件。

例如，`perf` 可以报告不佳的缓存局部性（poor cache locality）、大量的页错误（page faults）或活锁（livelocks）。下面是关于常见命令的简介：

- `perf list` - 列出可以被 pref 追踪的事件；
- `perf stat COMMAND ARG1 ARG2` - 收集与某个进程或指令相关的事件；
- `perf record COMMAND ARG1 ARG2` - 记录命令执行的采样信息并将统计数据储存在 `perf.data` 中；
- `perf report` - 格式化并打印 `perf.data` 中的数据。
