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

