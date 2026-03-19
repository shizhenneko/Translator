# 项目 2：Gitlet | CS 61B 2021 春季

## 元信息
- 来源: url https://sp21.datastructur.es/materials/proj/proj2/proj2
- 时间戳: 2026-02-27T15:39:17Z
- 模型: kimi-k2-0905-preview

## 大纲
### 项目 2：Gitlet 概述
- 摘要
  - 实现简化版分布式版本控制系统 Gitlet，支持提交、分支、合并等核心 Git 功能。
  - 项目依赖 Lab 6 经验与 Lecture 12 概念，需完成初始化、add、commit、log、checkout、branch、merge 等命令。
  - 所有数据持久化在 .gitlet 目录，使用 Java 序列化与 SHA-1 内容寻址。
  - 提供三段式自动评测：Checkpoint、Full Grader、Snaps，含 112 分额外加分。
  - 需提交设计文档，禁止共享代码，鼓励用 Ed 讨论与自写集成测试。
- 关键要点
  - 先完成 Lab 6 并观看官方视频，理解提交树、HEAD、分支、合并概念。
  - 按命令逐个实现，重用代码；merge 最复杂，勿拖到最后。
  - 利用提供的 tester.py 与 runner.py 编写 .in 测试，覆盖所有失败与危险场景。
  - 远程命令为额外加分，完成主干后再考虑。
### 内部结构与数据模型
- 摘要
  - Blob 存储文件内容；Commit 包含元数据、父引用、文件映射；Tree 被简化进 Commit。
  - 所有对象用 SHA-1 内容寻址，作为文件名存于 .gitlet，确保跨机器一致。
  - 提交树是不可变有向无环图；HEAD 与分支指针记录当前位置。
  - Java 序列化持久化对象，注意 transient 字段避免写入整个图。
- 关键要点
  - 用 SHA-1 字符串而非 Java 指针指代对象，避免序列化时拖入全图。
  - 统一工具类读写对象，确保哈希计算包含全部相关字段。
### 命令详细规范
- 摘要
  - init：创建含初始空提交的 .gitlet；add：单文件暂存；commit：生成新快照并清空暂存区。
  - log/global-log：打印历史；checkout 三种形式：文件还原、提交还原、分支切换。
  - branch/rm-branch：创建或删除分支指针；reset：强制回退分支头；merge：三路合并并处理冲突。
- 关键要点
  - 危险命令（checkout、reset、merge）需先检查未跟踪文件冲突。
  - 合并冲突文件用 <<<<<<< HEAD ======= >>>>>>> 格式标记，合并后自动提交。
### 测试与调试策略
- 摘要
  - 使用 testing/tester.py 编写 .in 集成测试，支持文件内容断言、正则匹配、变量捕获。
  - runner.py 单步执行调试；--keep 保留临时目录检查 .gitlet 内容。
  - 先本地通过样本测试，再有限次提交自动评测；设计文档必须随时更新。
- 关键要点
  - 每实现一个命令就写对应 .in 测试，利用 samples/*.inc 复用公共前置条件。
  - 调试时逐条命令验证磁盘状态，定位最早出错点。
### 远程命令（额外加分）
- 摘要
  - add-remote / rm-remote 管理远端别名；push / fetch / pull 实现快进式同步。
  - push 要求远端头在本地历史内；fetch 创建或更新 remote/branch 分支；pull = fetch + merge。
- 关键要点
  - 远程功能不评性能，先确保主干功能稳定再扩展。
## 术语表
| 英文术语 | 中文术语 | 说明 | 首次保留英文 |
| --- | --- | --- | --- |
| commit | 提交 | 保存项目快照的不可变节点，含文件映射、父引用、元数据。 | true |
| blob | 数据块 | 存储单个文件内容的对象，用 SHA-1 寻址。 | true |
| staging area | 暂存区 | 记录下次提交要添加或删除的文件，位于 .gitlet。 | true |
| branch | 分支 | 指向某提交的可移动引用，支持并行开发。 | true |
| HEAD | 头指针 | 标记当前分支与当前提交的引用。 | true |
| merge | 合并 | 把指定分支的修改并入当前分支，可能产生冲突文件。 | true |
| SHA-1 | SHA-1 哈希 | 160 位内容摘要，用作对象唯一 ID 与文件名。 | true |
| serialization | 序列化 | 将 Java 对象转为字节流以写入磁盘，反之为反序列化。 | true |
| working directory | 工作目录 | 用户可见的项目文件夹，不含 .gitlet。 | true |

项目 2：Gitlet | CS 61B Spring 2021
===============

*   [主页](https://sp21.datastructur.es/index.html)
*   [课程信息](https://sp21.datastructur.es/about.html)
*   [教师团队](https://sp21.datastructur.es/staff.html)
*   [资源](https://sp21.datastructur.es/resources.html)
*   [考试](https://sp21.datastructur.es/exams.html)
*   [Beacon](http://beacon.datastructur.es/)
*   [Ed](https://edstem.org/us/courses/3735/discussion/)
*   [OH 排队](https://oh.datastructur.es/)

*   [关于本规范](https://sp21.datastructur.es/materials/proj/proj2/proj2#a-note-on-this-spec)
*   [Gitlet 概述](https://sp21.datastructur.es/materials/proj/proj2/proj2#overview-of-gitlet)
*   [内部结构](https://sp21.datastructur.es/materials/proj/proj2/proj2#internal-structures)
*   [行为详细规范](https://sp21.datastructur.es/materials/proj/proj2/proj2#detailed-spec-of-behavior)
    *   [总体规范](https://sp21.datastructur.es/materials/proj/proj2/proj2#overall-spec)

*   [命令](https://sp21.datastructur.es/materials/proj/proj2/proj2#the-commands)
    *   [init](https://sp21.datastructur.es/materials/proj/proj2/proj2#init)
    *   [add](https://sp21.datastructur.es/materials/proj/proj2/proj2#add)
    *   [commit](https://sp21.datastructur.es/materials/proj/proj2/proj2#commit)
    *   [rm](https://sp21.datastructur.es/materials/proj/proj2/proj2#rm)
    *   [log](https://sp21.datastructur.es/materials/proj/proj2/proj2#log)
    *   [global-log](https://sp21.datastructur.es/materials/proj/proj2/proj2#global-log)
    *   [find](https://sp21.datastructur.es/materials/proj/proj2/proj2#find)
    *   [status](https://sp21.datastructur.es/materials/proj/proj2/proj2#status)
    *   [checkout](https://sp21.datastructur.es/materials/proj/proj2/proj2#checkout)
    *   [branch](https://sp21.datastructur.es/materials/proj/proj2/proj2#branch)
    *   [rm-branch](https://sp21.datastructur.es/materials/proj/proj2/proj2#rm-branch)
    *   [reset](https://sp21.datastructur.es/materials/proj/proj2/proj2#reset)
    *   [merge](https://sp21.datastructur.es/materials/proj/proj2/proj2#merge)

*   [骨架代码](https://sp21.datastructur.es/materials/proj/proj2/proj2#skeleton)
*   [设计文档](https://sp21.datastructur.es/materials/proj/proj2/proj2#design-document)
*   [评分细则](https://sp21.datastructur.es/materials/proj/proj2/proj2#grader-details)
    *   [阶段评分器](https://sp21.datastructur.es/materials/proj/proj2/proj2#checkpoint-grader)
    *   [完整评分器](https://sp21.datastructur.es/materials/proj/proj2/proj2#full-grader)
    *   [快照评分器](https://sp21.datastructur.es/materials/proj/proj2/proj2#snaps-grader)
    *   [额外加分](https://sp21.datastructur.es/materials/proj/proj2/proj2#extra-credit)

*   [项目须知杂项](https://sp21.datastructur.es/materials/proj/proj2/proj2#miscellaneous-things-to-know-about-the-project)
*   [文件处理](https://sp21.datastructur.es/materials/proj/proj2/proj2#dealing-with-files)
*   [序列化（serialization）细节](https://sp21.datastructur.es/materials/proj/proj2/proj2#serialization-details)
*   [测试](https://sp21.datastructur.es/materials/proj/proj2/proj2#testing)
*   [在官方解答上测试](https://sp21.datastructur.es/materials/proj/proj2/proj2#testing-on-the-staff-solution)
*   [理解集成测试](https://sp21.datastructur.es/materials/proj/proj2/proj2#understanding-integration-tests)
    *   [示例测试](https://sp21.datastructur.es/materials/proj/proj2/proj2#example-test)
    *   [测试准备](https://sp21.datastructur.es/materials/proj/proj2/proj2#setup-for-a-test)
    *   [输出模式匹配](https://sp21.datastructur.es/materials/proj/proj2/proj2#pattern-matching-output)
    *   [测试总结](https://sp21.datastructur.es/materials/proj/proj2/proj2#testing-conclusion)

*   [调试集成测试](https://sp21.datastructur.es/materials/proj/proj2/proj2#debugging-integration-tests)
    *   [定位待调试执行](https://sp21.datastructur.es/materials/proj/proj2/proj2#finding-the-right-execution-to-debug)

*   [远程操作（额外加分）](https://sp21.datastructur.es/materials/proj/proj2/proj2#going-remote-extra-credit)
*   [命令](https://sp21.datastructur.es/materials/proj/proj2/proj2#the-commands-1)
    *   [add-remote](https://sp21.datastructur.es/materials/proj/proj2/proj2#add-remote)
    *   [rm-remote](https://sp21.datastructur.es/materials/proj/proj2/proj2#rm-remote)
    *   [push](https://sp21.datastructur.es/materials/proj/proj2/proj2#push)
    *   [fetch](https://sp21.datastructur.es/materials/proj/proj2/proj2#fetch)
    *   [pull](https://sp21.datastructur.es/materials/proj/proj2/proj2#pull)

*   [I. 需避免事项](https://sp21.datastructur.es/materials/proj/proj2/proj2#i-things-to-avoid)
*   [J. 致谢](https://sp21.datastructur.es/materials/proj/proj2/proj2#j-acknowledgments)

项目 2：Gitlet
关于本规范
-------------------

本规范较长。前半部分用详尽文字描述你将支持的每条命令；后半部分讲解测试细节与建议。为便于消化，我们准备了大量优质视频，分段讲解规范并给出起步建议。所有视频已嵌入下文对应位置，也集中列在此处方便查看。注意：部分视频摄于 2020 春季，当时 Gitlet 是项目 3，Capers 是实验 12，并短暂提到 Hilfinger 教授的远程 `shared` 、仓库 `repo`  等，本学期可直接忽略，作业内容不变。

*   [Git 入门 - 第 1 部分](https://www.youtube.com/watch?v=yWBzCAY_5UI)
*   [Git 入门 - 第 2 部分](https://www.youtube.com/watch?v=CnMpARAOhFg)
*   [第 12 讲直播](https://youtu.be/fvhqn5PeU_Q)
*   Gitlet 入门播放列表
    *   [第 1 部分](https://www.youtube.com/watch?v=-1gE2cNFhPA)
    *   [第 2 部分](https://www.youtube.com/watch?v=GfmH9_8tM5w)
    *   [第 3 部分](https://www.youtube.com/watch?v=dv5VdbIZKF8)
    *   [第 4 部分](https://www.youtube.com/watch?v=k8jwbG8bE7Y)
    *   [Itai 所用幻灯片](https://cdn-uploads.piazza.com/attach/k5eevxebzpj25b/jqr7jm9igtc7l5/k97ipfmgmb3n/Gitlet_Slides.pdf)

*   [合并（merge）概览与示例](https://www.youtube.com/watch?v=JR3OYCMv9b4&t=929s)
*   [分支（branch）概览与示例](https://youtu.be/desB3AS6aZg)
*   [测试](https://www.youtube.com/watch?v=uMYpuQuHGu0&t=752s)
*   [设计持久化（手写笔记）](https://paper.dropbox.com/doc/Gitlet-Persistence--AyM0lOEaezWrTi7gG_Pt~bXcAg-zEnTGJhtUMtGr8ILYhoab)
*   2021 春季 Office Hours 演示：
    *   Gitlet 起步
        *   [第 1 部分](https://youtu.be/6JVdbNZm0cM)
        *   [第 2 部分](https://youtu.be/1d1yOSoTVAM)

    *   [设计 Gitlet](https://youtu.be/G3YU9oY8PcU)
        *   [笔记](https://sp21.datastructur.es/materials/proj/proj2/gitlet-design-notes.pdf)

    *   [合并](https://youtu.be/l0X5NgzAWYQ)

后续若有新资源，会在此更新，记得常刷新！

Gitlet 概述
------------------

**警告：** 务必先完成 [实验 6：Canine Capers](https://sp21.datastructur.es/materials/lab/lab6/lab6)。该实验是本项目的引子，能帮你顺利起步并确认环境就绪。同时请观看 [第 12 讲：Gitlet](https://youtu.be/fvhqn5PeU_Q)，其中介绍了对本项目极有帮助的核心思路。

在本项目中，你将实现一个**版本控制系统（version-control system）**，模仿流行系统 Git 的部分基础功能。我们的版本更轻量、更简单，因此取名 Gitlet。

> **学习批注：** Gitlet 只实现“本地”操作；后续“远程”部分为额外加分项。先聚焦本地 `commit`、`branch`、`merge` 三条主线，再考虑远程命令。

版本控制系统本质上就是“相关文件集合的备份系统”。Gitlet 支持的核心功能如下：

1. 保存整个目录的文件内容。在 Gitlet 中这叫**提交（commit）**，保存下来的内容称为**提交（commit）**。
2. 恢复一个或多个文件或整个提交的历史版本。在 Gitlet 中称为**检出（checkout）**这些文件或该提交。
3. 查看备份历史。在 Gitlet 中通过**日志（log）**查看。
4. 维护相关的提交序列，称为**分支（branch）**。
5. 把一个分支的更改**合并（merge）**到另一个分支。

> **学习批注：** 把版本控制想成“无限撤销+时光机”：每次 `commit` 就是给项目拍一张不可修改的照片，任何时候都能回到旧照片。

版本控制的意义在于：当你做复杂项目（或与他人协作）时，可以定期保存项目状态；若日后代码出错，可恢复到之前的提交，同时不会丢失之后的更改。协作者的提交也可被合并到你的版本。

Gitlet 并非逐文件提交，而是一次提交**整个项目快照**。下文示例虽常只改一个文件，但请记住：一次提交可包含对多个文件的修改。

可视化提交历史有助于理解。假设项目只有 `wug.txt`，我们依次修改并提交三次，就得到三个版本，可画成：

![Image 1: Three commits](https://sp21.datastructur.es/materials/proj/proj2/image/three_commits.png)

箭头表示每个提交指向它的**父提交（parent commit）**——这其实就是**链表**结构。

> **背景扩展：** 真正的 Git 内部对象也是靠 SHA-1 哈希把“节点”串成图；Gitlet 简化为单目录，因此更像一条链表。

如果想回到第 2 个提交的状态，只需让 Gitlet 找到链表第 2 个节点，把文件恢复到那时的样子，同时删除节点 1 有而节点 2 没有的文件。但此时链表“前端”不再反映当前文件状态，容易误导。为此引入**头指针（HEAD pointer）**，标记我们正处在链表的哪个节点。

正常提交时，HEAD 始终指向链表最前端：

![Image 2: Simple head](https://sp21.datastructur.es/materials/proj/proj2/image/simple_head.png)

若执行**重置（reset）**回到旧提交，HEAD 也随之移动：

![Image 3: Reverted head](https://sp21.datastructur.es/materials/proj/proj2/image/reverted_head.png)

这时就处于**游离头指针状态（detached HEAD）**——在标准 Git 里很常见。

> **学习批注：** 游离头指针=“当前不在任何分支尖端”，此时新提交容易“丢失”，因为没分支引用指向它。

EDITED 3/5: 注意 Gitlet 无法进入游离头指针状态，因为没有 `checkout`  命令能把 HEAD 直接移到某提交； `reset`  命令虽会移动 HEAD，但也同时移动分支指针，因此 Gitlet 始终有分支“托底”。

仅靠线性链表还不够酷。Gitlet 还能保存**不同版本**：假设你对项目有 A、B 两套方案，可分别保存并随时切换。图示如下：

![Image 4: Two versions](https://sp21.datastructur.es/materials/proj/proj2/image/two_versions.png)

结构已从链表进化成**提交树（commit tree）**。每个独立版本称为树的一个**分支（branch）**，可并行开发：

![Image 5: Two developed versions](https://sp21.datastructur.es/materials/proj/proj2/image/two_developed_versions.png)

树中有两个指针，分别指向各分支的最新提交；当前活跃的那个叫**头指针（HEAD）**。HEAD 永远位于当前分支的“最前端”。

以上就是 Gitlet 的极简鸟瞰图！若暂时没完全看懂也无妨，下文会有详细规范。

最后强调：**提交树是不可变（immutable）**的——节点一旦创建就不能修改或删除，只能追加。这是 Gitlet 的设计底线，防止意外丢失历史。

---

内部结构
--------

真正的 Git 区分多种**对象（object）**。我们关心三类：

- **数据块（blob）**：文件内容的快照。同一文件在不同提交中可能对应多个 blob。
- **树（tree）**：目录结构，把文件名映射到 blob 或其他 tree（子目录）。
- **提交（commit）**：包含日志消息、元数据（时间、作者等）、指向 tree 的引用、指向父提交的引用。仓库还维护“分支头→提交”的映射，让重要提交有符号名。

Gitlet 进一步简化：

- 把 tree 并入 commit，且不处理子目录，每个仓库只有一层“扁平”普通文件。
- 合并（merge）只涉及两个父提交（真实 Git 支持任意数量）。
- 元数据仅保留时间戳与日志消息。因此一个 commit 包含：日志、时间戳、文件名→blob 引用映射、父引用、（合并时）第二父引用。

> **学习批注：** 由于无子目录，Gitlet 的“tree”概念被隐藏，你只需关心文件→blob→commit 的映射关系即可。

每个对象——在我们的场景里就是每个 `blob` 和每个 `commit`——都有一个唯一的整数 ID，用作对象的引用。Git 的一个有趣特性是这些 ID 是**通用（universal）**的：与典型的 Java 实现不同，两个内容完全相同的对象在所有系统上都会拥有相同的 ID（也就是说，我的电脑、你的电脑、任何其他人的电脑都会算出完全一样的 ID）。对于 `blob` 而言，“相同内容”指文件内容一致；对于 `commit`，则指元数据、文件名到引用的映射、以及父引用都相同。于是，仓库里的对象被称为**内容可寻址（content addressable）**。

Git 与 Gitlet 都用同一种方式实现这一点：借助名为 SHA-1（Secure Hash 1）的**加密哈希函数（cryptographic hash function）**，它能把任意字节序列映射成 160 位整数哈希。加密哈希函数的特点是：想找出两段不同字节流却拥有相同哈希值，几乎不可能（甚至仅给定哈希值，想反推字节流也几乎办不到）。因此，我们可以假定两个内容不同的对象出现 SHA-1 碰撞的概率是 2⁻¹⁶⁰，约 10⁻⁴⁸。我们干脆忽略碰撞可能——理论上系统存在“根本缺陷”，但实践中永远不会发生！

好在已有库类帮你计算 SHA-1，无需自己实现算法。你要做的只是确保给对象正确“贴标签”。具体包括：

*   对 `commit` 做哈希时，要把所有元数据和引用都算进去。  
*   区分 `commit` 哈希与 `blob` 哈希。一个办法是在 `.gitlet`  目录里设计合理的子目录结构；另一个办法是给每类对象额外塞一个单词再哈希，比如 `blob` 加一个前缀，`commit` 加另一个。

顺带一提，SHA-1 哈希值写成 40 位十六进制字符串，正好当文件名，用来把数据存进 `.gitlet`  目录（下文详述）。比较两个文件（`blob`）是否相同也超方便：SHA-1 一样就当它们一样。

远程（remote）方面，我们直接用其它 Gitlet 仓库（就像整学期都在用的 `skeleton` ）。`push` 就是把远端还没有的 `commit` 和 `blob` 复制过去，并更新分支指针；`pull` 则是反方向操作。远程功能属附加分，不拿也能满分。

读写内部对象到文件其实很简单，多亏 Java 的**序列化（serialization）**机制。接口 `java.io.Serializable`  虽然空方法都没有，但只要类实现了它，Java 运行时就自动提供对象与字节流的双向转换；接着用 I/O 类 `java.io.ObjectOutputStream`  把字节写文件，再用 `java.io.ObjectInputStream`  读回并反序列化即可。“序列化”就是把任意结构（数组、树、图等）转成连续字节序列。实验 6 你已练过，这里做法几乎一样，遇到持久化问题直接翻 lab6 代码即可。

下面给出本节结构的汇总示例：每个 `commit`（矩形）指向若干 `blob`（圆形），`blob` 存文件内容；`commit` 存文件名到 `blob` 的映射，还有父链接。这些引用（箭头）在 `.gitlet`  目录里用 SHA-1 哈希值表示（十六进制小字）。新 `commit` 更新了 `wug1.txt` ，却与旧 `commit` 共享同版 `wug2.txt` 。你的 `commit` 类得把图里所有信息都存下来——内部数据结构选得好，实现就轻松；选得差，就头秃。先花点时间规划！

![图 6：两个提交及其数据块](https://sp21.datastructur.es/materials/proj/proj2/image/commits-and-blobs.png)

行为详细规范
-------------------------
#### 整体规范

我们唯一强制要求的结构是：必须有一个名为 `gitlet.Main`  的类，且它包含 `main` 方法。

我们还提供了一些工具方法，帮你完成大部分与文件系统相关的杂活，这样你就能把精力集中在项目逻辑上，而不是跟操作系统较劲。

另外，我们给了两个“建议类”： `Commit`  和 `Repository` ，供你起步。你可以随意增删 Java 类，但**禁止引入外部代码**（JUnit 除外），也**只能用 Java** 写。Java 标准库随便用，我们提供的工具也能用。

**别把所有逻辑都塞进 Main。**`Main` 应该只是调 `Repository`  里的助手方法。参考实验 6 的 `CapersRepository`  和 `Main`  类，那就是我们推荐的结构。

本规范的大部分内容会逐条说明：当 `Gitlet.java`  的 `main` 收到不同 gitlet 命令时，该怎么反应。先看几条全局约束：

*   Gitlet 得有个地方存旧版文件和元数据。**必须**放在名为 `.gitlet`  的目录里，就像真 git 把信息存在 `.git`  目录一样（带 `.`  前缀的是隐藏文件，默认看不到；Unix 下用 `ls -a`  就能列出）。只要某目录下有 `.gitlet` ，就认为 Gitlet 在此“已初始化”。除 `init`  命令外，其余命令都只能在已初始化的目录里运行——也就是当前目录得包含 `.gitlet` 。那些**不在** `.gitlet`  目录里的文件（即你正在编辑、或打算加入仓库的文件）统称**工作目录（working directory）**里的文件。

*   多数命令对运行时间或内存有限制。其中“相对于任何显著度量”的常数时间要求，显著度量指：文件数量、文件大小、提交（commit）数量。序列化（serialization）/反序列化时间可忽略，**但序列化时间不能依赖已添加或已提交文件的总大小**（忘了啥是序列化？回去翻实验 6）。哈希表查找可视为常数时间。

*   部分命令有指定的失败场景与错误信息，后文会给出格式。所有错误信息以句点结尾；自动评分会逐字比对，别漏点。遇到失败场景，**只打印错误信息，别动其他状态**。规范里没列出的错误场景，你无需处理。

*   还有几条通用失败场景：

    *   用户没给任何参数 → 打印 `Please enter a command.`  并退出。
    *   用户输入了不存在的命令 → 打印 `No command with that name exists.`  并退出。
    *   用户输入的命令参数数量或格式不对 → 打印 `Incorrect operands.`  并退出。
    *   用户在某条命令需要已初始化 Gitlet 工作目录（即含 `.gitlet` 子目录）时，却不在这样的目录里 → 打印 `Not in an initialized Gitlet directory.` 并退出。

*   部分命令会与真 git 有差异，下文会特别指出。规范不会穷尽所有差异，但会把容易混淆的大坑标出来。

*   **除规范明确要求外，别打印任何额外内容。**多印一个字符都可能让自动评分爆炸。

*   想立即退出程序，可调用 `System.exit(0)` 。例如助手函数里遇到错误，希望 gitlet 立刻终止，就调它。**注意：必须传参数 0**。在 61C 你会学到这个参数（错误码）的含义。

*   规范把某些命令标为“危险”。危险命令可能覆盖非元数据文件——比如让用户把文件恢复到旧版本，就会覆盖当前版本。友情提示：测试这类命令前，先戴头盔 :)

命令详解
------------

下面逐条拆解你必须支持的命令。好程序员永远先想数据结构：读命令时，先琢磨“该咋存数据才能秒支持这条命令”，再想想“能不能复用前面写过的代码”（项目 2 后期大量复用前期代码， hint 已经给足）。部分方法旁我们标注了推荐讲座，仅供参考。更绕的命令配有概念小测，**不计分**，纯帮你自测理解，动手前务必刷一遍。
#### init

*   **用法**：  
    ```plaintext
    java gitlet.Main init
    ```

*   **描述**：在当前目录新建一个 Gitlet 版本控制系统。系统会自动生成一个初始提交（commit）：该提交不含任何文件，提交信息为 `initial commit` （无标点）。系统仅有一个分支（branch） `master` ，它指向该初始提交，且 `master`  为当前分支。初始提交的时间戳为 1970-01-01 00:00:00 UTC（即 Unix 纪元，内部用时间 0 表示）。由于所有 Gitlet 仓库的初始提交内容完全相同，因此它们共享同一个 UID，所有后续提交都可追溯至此提交。

> **学习批注：** 可以把初始提交想象成“宇宙大爆炸”，所有历史都从这里开始，且不同仓库的“大爆炸”是同一次事件。

*   **运行时间**：与任何显著度量相比应为常数级。

*   **失败情况**：若当前目录已存在 Gitlet 系统，应中断并打印错误信息
    ```plaintext
    A Gitlet version-control system already exists in the
    current directory.
    ```
，**不得**覆盖已有系统。

*   **危险？**：否

*   **我们的代码行数**：约 15 行

#### add

*   **用法**：  
    ```plaintext
    java gitlet.Main add [file name]
    ```

*   **描述**：把文件当前状态的副本加入**暂存区（staging area）**（详见 `commit`  命令说明）。因此，添加文件也叫“将文件**暂存（stage）**以备**提交（commit）**”。若文件已暂存，新内容会覆盖旧条目。暂存区应放在 `.gitlet`  内。若工作目录（working directory）中的文件与当前提交版本完全一致，则不再暂存，并把它从暂存区移除（可能先改、再暂存、又改回原样的情况）。若该文件当时被标记为“待删除”，也会被取消（见 `gitlet rm` ）。

> **学习批注：** 暂存区就像购物车，只决定“结账”时带什么；重复添加同一文件相当于把购物车里的旧货换成新货。

*   **运行时间**：最坏情况下与文件大小成线性关系，并与提交中文件数 N 的 lg N 成正比。

*   **失败情况**：文件不存在则打印错误 `File does not exist.`  并直接退出，不做任何修改。

*   **危险？**：否

*   **我们的代码行数**：约 20 行

*   **与真实 git 的差异**：真实 git 可一次添加多个文件；gitlet 一次只能添加一个。

*   **推荐复习课程**：第 16 讲（集合、映射、ADT）、第 19 讲（哈希）

#### 提交（commit）

*   **用法**：  
    ```plaintext
    java gitlet.Main commit [message]
    ```


*   **描述**：将当前提交（commit）和暂存区（staging area）里已跟踪文件的快照保存下来，以便以后恢复，并生成一个新的提交。该提交会“跟踪”这些被保存的文件。默认情况下，每个提交的文件快照与其父提交完全一致；文件版本不会被更新，除非该文件被加入暂存区等待添加。此时，提交将包含暂存区的版本，而不再使用父提交中的版本。若某些文件被加入暂存区但父提交并未跟踪它们，提交会开始跟踪这些文件。反之，若当前提交中的文件被 `rm`  命令标记为“暂存移除”，则在新提交中它们将不再被跟踪。

一句话总结：默认情况下，提交与父提交文件内容相同；只有“暂存添加”和“暂存移除”的文件才会带来变化。当然，日期（以及日志信息）通常也会不同。

关于提交的补充要点：

    *   提交后暂存区会被清空。  
    *   提交命令不会在工作目录（working directory）里增删改文件（ `.gitlet`  目录内的除外）。`rm` 命令会同时从工作目录删除文件并暂存移除，使得 `commit`  后它们不再被跟踪。  
    *   一旦文件被暂存添加或移除，之后对其的任何改动都会被 `commit`  命令忽略；该命令只操作 `.gitlet`  目录。例如，用 Unix 的 `rm`  命令删除一个已跟踪文件，不会影响下一次提交，提交仍包含该文件（尽管工作目录已删）。  
    *   提交后，新节点加入提交树。  
    *   刚完成的提交成为“当前提交”，HEAD 指向它；原 HEAD 提交成为其父提交。  
    *   每个提交应记录生成时的日期与时间。  
    *   每个提交都附带一条日志信息，描述文件变更，由用户指定。整条信息在传给 `main`  的数组 `args`  中只占一项；多词信息需用引号包裹。  
    *   每个提交由 SHA-1 哈希唯一标识，需包含文件（数据块 blob）引用、父提交引用、日志信息与提交时间。

> **学习批注：** 可以把提交想象成拍照：默认复制上一张照片，只把“摆好姿势”（暂存）的人换掉。

*   **运行时间**：与提交数量无关，应为常数；与所跟踪文件总大小成线性或更优。空间要求：提交后 `.gitlet`  目录增量不得超过“暂存添加”文件总大小（不含额外元数据）。提示：数据块按内容寻址，利用 SHA-1 避免重复存储。允许保存完整文件副本，无需仅存差异。

*   **失败情况**：若无文件被暂存，则中止并打印
    ```plaintext
    No
    changes added to the commit.
    ```

    提交信息不能为空，否则打印
    ```plaintext
    Please enter
    a commit message.
    ```
    已跟踪文件在工作目录缺失或改动不算失败，完全忽略 `.gitlet`  目录外内容。

*   **危险？**：否

*   **与真实 git 差异**：真实 git 的提交可有多个父提交（合并），且元数据更丰富。

*   **我们的代码行数**：约 35 行

*   **建议先修讲座**：第 19 讲（集合、映射、ADT）、第 19 讲（哈希）

提交前后示意图：

![图 7：提交前后对比](https://sp21.datastructur.es/materials/proj/proj2/image/before_and_after_commit.png)
#### rm

*   **用法**：  
    ```plaintext
    java gitlet.Main rm [file name]
    ```

*   **说明**：如果文件已暂存（staged）等待添加，则将其移出暂存区；若文件在当前提交（commit）中被跟踪（tracked），则将其标记为待删除，并把它从工作目录（working directory）中移除（仅当该文件确实被当前提交跟踪时才删除）。  
  > **学习批注：** “stage for removal” 相当于 Git 的 `git rm`，既删文件又记删除操作；若文件未被跟踪，直接报错。

*   **运行时间**：与任何重要指标相比应为常数时间。

*   **失败情形**：若文件既未暂存，也未被 HEAD 提交跟踪，打印错误信息  
    ```plaintext
    No reason to remove the file.
    ```

*   **危险？**：是（但使用我们提供的工具方法时，只会破坏仓库文件，不会影响目录中的其他文件）。

*   **我们的代码行数**：约 20 行

#### log

*   **用法**：  
    ```plaintext
    java gitlet.Main log
    ```


*   **描述**：从当前头指针（HEAD）提交（commit）出发，沿着提交树向回遍历，直到初始提交，仅追踪每个合并提交（merge commit）的第一个父提交，忽略第二个父提交。（在常规 Git 中，这相当于 `git log --first-parent` 。）这组提交节点称为该提交的**历史（history）**。对历史中的每个节点，需显示：提交 id、提交时间、提交消息。必须严格遵循以下格式：

    ```
    ===
    commit a0da1ea5a15ab613bf9961fd86f010cf74c7ee48
    Date: Thu Nov 9 20:00:05 2017 -0800
    A commit message.

    ===
    commit 3e8bf1d794ca2e9ef8a4007275acf3751c7170ff
    Date: Thu Nov 9 17:01:33 2017 -0800
    Another commit message.

    ===
    commit e881c9575d180a215d1a636545b8fd9abfb1d2bb
    Date: Wed Dec 31 16:00:00 1969 -0800
    initial commit
    ```


每条提交前有一个 `===` ，提交后留一空行。与真实 Git 一样，每条记录展示提交对象唯一的 SHA-1 哈希（SHA-1）。时间戳按当前时区显示，而非 UTC；因此初始提交不会显示“1970-01-01 00:00:00”，而是对应的太平洋标准时间。你所在时区可能不同，这没关系。

提交按“最新在上”顺序列出。顺便提醒，Java 的 `java.util.Date`  与 `java.util.Formatter`  类可方便地获取并格式化时间，别手动拼字符串！

当然，SHA-1 标识符会与示例不同，无需担心。测试会确保你输出“看起来像”SHA-1 的字符串（详见后文测试部分）。

对于合并提交（有两个父提交），在第一条下方追加一行，如

```
===
commit 3e8bf1d794ca2e9ef8a4007275acf3751c7170ff
Merge: 4975af1 2c1ead1
Date: Sat Nov 11 12:30:00 2017 -0800
Merged development into master.
```


其中 “Merge:” 后的两个十六进制数分别是第一、第二父提交 id 的前 7 位。第一父提交是执行合并时所在分支（branch）的提交，第二父提交是被合并分支的提交。这与常规 Git 一致。

*   **运行时间**：与头指针历史中节点数成线性关系。

*   **失败情况**：无

*   **危险操作？**：否

*   **我们的行数**：约 20 行

下图展示某提交的历史。若当前分支的头指针指向该提交，log 将打印出被圈出的提交信息：

![Image 8: History](https://sp21.datastructur.es/materials/proj/proj2/image/history.png)

历史忽略其他分支及“未来”提交。既然引入了历史概念，我们可更精确地重申：提交树之所以不可变，是因为**某个特定 id 的提交历史永远不变**。若把提交树视为若干历史的集合，则每条历史本身都是不可变的。

> **学习批注：**  
> 可以把“历史”想象成单向链表：每个节点只指向父节点，永不回头修改。这样设计保证了即使之后创建新分支或合并，旧提交的快照与关系仍纹丝不动，天然支持可追溯性与协作安全。
#### 全局日志（global-log）

*   **用法**：  
    ```plaintext
    java gitlet.Main global-log
    ```

*   **描述**：类似 `log`，但显示**所有**曾经创建的提交（commit）。提交顺序任意。提示：在 `gitlet.Utils`  里有一个实用方法，可帮你遍历目录下的文件。

*   **运行时间**：与总提交数成线性关系。

*   **失败场景**：无

*   **危险操作？**：否

*   **我们的代码行数**：约 10 行

> **学习批注：** 实现时只需遍历 `.gitlet/commits` 目录下所有序列化文件，逐个反序列化并打印即可。
#### find

*   **用法**：  
    ```plaintext
    java gitlet.Main find [commit message]
    ```


*   **描述**：打印出所有具有指定提交（commit）信息的提交 id，每行一个。若存在多条，分行输出。提交信息为单个操作数；多词信息请用引号包裹，与下方 `commit`  命令用法一致。提示：该命令的提示与 `global-log`  相同。

*   **运行时间**：应与提交数量成线性关系。

*   **失败情况**：若找不到对应提交，打印错误信息  
    ```plaintext
    Found no commit with that message.
    ```


*   **危险？**：否

*   **与真实 git 的差异**：真实 git 无此命令，可通过 `git log | grep` 实现类似效果。

*   **我们的代码行数**：约 15 行

> **学习批注：** 该命令本质是遍历所有提交对象，比对 `message` 字段，类似在文件系统里用 `grep` 搜关键字。
#### status

*   **用法**：  
    ```plaintext
    java gitlet.Main status
    ```

*   **描述**：显示当前存在的所有`分支（branch）`，并用 `*` 标出当前分支。同时列出已暂存（staged）待添加或删除的文件。_必须_按以下格式输出：

    ```
    === Branches ===
    *master
    other-branch

    === Staged Files ===
    wug.txt
    wug2.txt

    === Removed Files ===
    goodbye.txt

    === Modifications Not Staged For Commit ===
    junk.txt (deleted)
    wug3.txt (modified)

    === Untracked Files ===
    random.stuff
    ```  
  
最后两部分（“未暂存修改”与“未跟踪文件”）为额外加分项，共 32 分，可留空（仅保留标题即可）。

各节之间空一行，末尾再空一行。条目按 Java 字符串字典序排列（星号不参与排序）。若文件满足以下任一情况，则视为“已修改但未暂存”：

    *   当前`提交（commit）`已跟踪，但`工作目录（working directory）`内容变动且未暂存；或  
    *   已暂存待添加，但`工作目录`内容与暂存区不同；或  
    *   已暂存待添加，却在`工作目录`中被删除；或  
    *   未暂存待删除，但在当前`提交`中跟踪且已从`工作目录`删除。

最后一类“未跟踪文件”指：存在于`工作目录`，但既未暂存添加，也未被跟踪。包括“先暂存删除、后又偷偷新建”的文件。忽略所有子目录，Gitlet 不处理它们。

> **学习批注：** 把 status 想成“体检报告”：分支是“科室”，文件是“指标”，星号告诉你正在哪个科室看病。

*   **运行时间**：仅依赖`工作目录`数据量 + 暂存文件数 + 分支数。

*   **失败情况**：无

*   **危险操作？**：否

*   **参考行数**：约 45 行

*   [**概念测验（不含分支）**](https://forms.gle/LSgBK5RAdRwhAqKK8)

*   [**概念测验（含分支）**](https://forms.gle/RHUiRkSrtgysC6En8)
#### checkout

`checkout` 是一个通用命令，根据参数不同能做三件事。下面 3 段分别对应 3 种用法。

*   **用法**：

1. `java gitlet.Main checkout -- [file name]`
2. `java gitlet.Main checkout [commit id] -- [file name]`
3. `java gitlet.Main checkout [branch name]`

*   **描述**：

    1.   取出当前头提交（head commit）中的该文件版本，写回工作目录（working directory）；若该文件已存在则覆盖。写回后的新版本不会自动加入暂存区（staged）。

    2.   取出给定提交 id 对应提交中的该文件版本，写回工作目录；若已存在则覆盖。写回后的新版本不会自动加入暂存区。

    3.   取出给定分支头提交里的全部文件并写回工作目录；若文件已存在则覆盖。命令结束后，该分支会成为当前分支（HEAD）。当前分支中被跟踪但目标分支中不存在的文件会被删除。除“检出的就是当前分支”这一失败情况外，暂存区会被清空（见下文**失败情况**）。

*   **运行时间**：

    1.   与被检出文件大小线性相关。

    2.   与提交快照中文件总大小线性相关；相对于提交数量应为常数；相对于分支数量也应为常数。

*   **失败情况**：

前一个提交中不存在该文件时，终止并打印：
```plaintext
File does not exist in that
    commit.
```
不改变当前工作目录（CWD）。

给定 id 的提交不存在时，打印：
```plaintext
No commit with
    that id exists.
```
若该提交中无此文件，也打印同样信息。不改变 CWD。

指定分支不存在时，打印 `No such branch exists.`。

该分支就是当前分支时，打印：
```plaintext
No need to checkout the
current branch.
```

若工作目录中有未跟踪文件会被覆盖，打印：
```plaintext
There is an untracked file in the way; delete it, or add and commit it
first.
```
并立即退出；该检查最先执行。不改变 CWD。

*   **与真实 git 的差异**：真实 git 不会清空暂存区（staging area），还会把被检出文件加入暂存区；此外，它拒绝覆盖已暂存的修改（新增或删除）。

> **学习批注：** Gitlet 的 checkout 直接丢弃暂存区内容，而真实 git 会保留并更新暂存区，这是两者行为差异的关键点。

一个 `[commit id]`  是前面提到的十六进制数。真实 Git 支持用唯一前缀缩写提交号，例如可把

```
a0da1ea5a15ab613bf9961fd86f010cf74c7ee48
```

简写成

```
a0da1e
```

（只要前 6 位不与其他 SHA-1 哈希冲突）。你也需实现相同的前缀匹配。若实现粗暴，查找时间会与对象数量成线性关系，因此不用在意性能。建议观察 `.git`  目录（特别是 `.git/objects` ）如何利用文件系统加速搜索，你会认出某种“用目录树代替指针”的经典数据结构。

只有第 3 种用法（切换分支）会修改暂存区；其他两种用法保留已计划的添加或删除。

*   **危险？**：是！

*   **我们写的行数**：

    *   ~15
    *   ~5
    *   ~15

*   [**概念测验（无分支）**](https://forms.gle/mfHLnrU9VX349jnr7)

*   [**概念测验（含分支）**](https://forms.gle/tbZuqDz7x3u41JhM6)
#### 分支（branch）

*   **用法**：  
    ```plaintext
    java gitlet.Main branch [branch name]
    ```


*   **描述**：以给定名称新建一个分支（branch），并让它指向当前的 `HEAD` 提交。分支本质上只是一个名字，对应某个提交节点的 SHA-1 哈希（SHA-1 hash）引用。此命令**不会**立即切换到新分支（与真实 Git 行为一致）。在首次调用 `branch` 前，你的代码应默认运行在名为 “master” 的分支上。

*   **运行时间**：应与任何重要指标成常数关系。

*   **失败情况**：若已存在同名分支，打印错误信息  
    ```plaintext
    A branch with that name already exists.
    ```


*   **危险？**：否

*   **我们的代码行数**：约 10 行

好，我们来看分支到底做了什么。假设当前状态如下：

![图 9：简单历史](https://sp21.datastructur.es/materials/proj/proj2/image/simple_history.png)

现在执行 `java gitlet.Main branch cool-beans` ，结果：

![图 10：刚创建分支](https://sp21.datastructur.es/materials/proj/proj2/image/just_called_branch.png)

嗯……好像什么都没变。再用  
```plaintext
java gitlet.Main
checkout cool-beans
```
 切换到这个分支：

![图 11：刚切换分支](https://sp21.datastructur.es/materials/proj/proj2/image/just_switched_branch.png)

又没变化？！好，那我们现在做一次提交（commit）。先改些文件，然后 `java gitlet.Main add...`  接着  
```plaintext
java gitlet.Main commit...
```
  
![图 12：在分支上提交](https://sp21.datastructur.es/materials/proj/proj2/image/commit_on_branch.png)

不是说好会分叉吗？怎么还是一条直线？那我切回另一个分支试试，用  
```plaintext
java
gitlet.Main checkout master
```
：

![图 13：检出 master](https://sp21.datastructur.es/materials/proj/proj2/image/checkout_master.png)

现在再做一次提交……

![图 14：终于分叉](https://sp21.datastructur.es/materials/proj/proj2/image/branched.png)

呼！这就是分支的核心思想。你看出门道了吗？创建分支只是多给你一个指针。任何时候，只有一个指针被认为是“当前活跃”的，也就是 HEAD 指针（用 * 标记）。我们可以用 `checkout [branch name]`  来回切换 HEAD。每当你提交时，就在当前 HEAD 指向的提交下再添加一个子提交，即使该提交已有其他子提交，也会自然形成分叉。

分支的视频示例与概览见[此处](https://youtu.be/desB3AS6aZg)。

务必确保你的 `branch` 、 `checkout`  和 `commit`  行为与上述一致。这是 Gitlet 的核心功能，许多其他命令都依赖它。一旦核心逻辑出错，大量自动评分测试将无法通过！

> **学习批注：** 分支只是“贴标签的便签”，贴在哪就在哪；HEAD 像手指，指哪打哪。先贴标签再移动手指，才能真正“分叉”。
#### rm-branch

*   **用法**：  
    ```plaintext
    java gitlet.Main rm-branch [branch name]
    ```


*   **描述**：删除指定名称的 分支（branch）。仅删除该分支的指针，不会删除在该分支下创建的所有提交或其他内容。

*   **运行时间**：应与任何重要度量成常数关系。

*   **失败情况**：若指定名称的分支不存在，则中止操作，并打印错误信息  
    ```plaintext
    A branch with that name does not
    exist.
    ```
  
若尝试删除当前所在分支，则中止操作，并打印错误信息  
```plaintext
Cannot remove the current branch.
```


*   **危险？**：否

*   **我们的代码行数**：约 15 行

> **学习批注：** 删除分支就像撕掉一本书的标签，书页（提交）仍在，只是标签没了。  
> **背景扩展：** Git 的分支本质上是指向提交的轻量指针，因此删除分支不会触发提交回收。
#### reset

*   **用法**：  
    ```plaintext
    java gitlet.Main reset [commit id]
    ```


*   **描述**：检出指定提交（commit）所追踪的所有文件，并删除当前分支中不在该提交里的已追踪文件。同时将当前分支的 `HEAD`（头指针）移动到该提交节点。关于 `HEAD` 移动后的效果，见课程开头的图示。 `[commit id]`  可以像 `checkout`  那样缩写。执行后暂存区（staging area）被清空。该命令本质上是“切换到任意提交”的 `checkout` ，并顺带把分支头也挪过去。

> **学习批注：** 可以把 `reset` 想象成“时光机+强制整理”：既把文件回退到旧版本，也把分支指针硬拉到那个旧节点，之后的新提交会从这个旧点继续生长。

*   **运行时间**：与目标提交快照中所有文件的总大小呈线性关系；与提交数量无关，应为常数级。

*   **失败场景**：  
  - 若找不到对应 id 的提交，打印
      ```plaintext
      No
      commit with that id exists.
      ```
。  
  - 若工作目录（working directory）中有未追踪文件会被重置覆盖，先打印 `There is an untracked file in the way; delete it, or add and commit it first.` 并立即退出；该检查在所有操作前完成。

*   **危险？**：是！

*   **与真实 Git 的差异**：最接近真实 Git 中带 `--hard`  选项的用法，例如
    ```plaintext
    git reset --hard [commit
    hash]
    ```
。

*   **我们的行数**：约 10 行。怎么这么少？记得复用已有代码 :)
#### merge

*   **用法**：  
    ```plaintext
    java gitlet.Main merge [branch name]
    ```


*   **描述**：将给定分支的文件并入当前分支。这一步较复杂，详细流程如下：

1. 先找到当前分支与给定分支的**分离点（split point）**。例如，若 `master` 是当前分支，`branch` 是给定分支：  
   ![Image 15: Split point](https://sp21.datastructur.es/materials/proj/proj2/image/split_point.png)  
   分离点是两条分支头节点**最近的共同祖先（latest common ancestor）**：  
   - **共同祖先**：存在一条路径（0 条或多条父指针）从两个分支头都能到达的提交。  
   - **最近**：该共同祖先不能是其他共同祖先的祖先。  
   若分离点恰好是给定分支的提交，则直接结束，打印
   ```plaintext
   Given branch is an ancestor of the current branch.
   ```
   若分离点就是当前分支头，则效果等同于切换分支，打印
   ```plaintext
   Current branch fast-forwarded.
   ```
   否则继续以下步骤。

2. 给定分支自分离点后**修改过**、当前分支未改的文件，更新为给定分支版本并自动加入暂存区。  
   > **学习批注：** “修改”指文件内容在分离点与分支头之间发生变化；blob 按内容寻址，比较 SHA-1 即可判断。

3. 仅当前分支修改过的文件保持原样。

4. 两个分支以**完全相同方式**修改的文件（内容一致或同时删除）保持不变。若两边都删了，但工作目录里还有同名文件，也保持未跟踪状态。

5. 分离点不存在、仅出现在当前分支的新文件保留。

6. 分离点不存在、仅出现在给定分支的新文件检出并暂存。

7. 分离点存在、当前分支未改、给定分支已删除的文件，从工作目录移除并取消跟踪。

8. 分离点存在、给定分支未改、当前分支已删除的文件，继续缺席。

9. **冲突文件**：两边修改方式不同（内容不同，或一边改一边删，或分离点不存在但内容不同）。将冲突文件内容替换为
   ```plaintext
   <<<<<<< HEAD
   contents of file in current branch
   =======
   contents of file in given branch
   >>>>>>>
   ```
   并暂存。删除视为空文件，直接拼接。若文件末尾无换行，可能出现
   ```plaintext
   <<<<<<< HEAD
   contents of file in current branch=======
   contents of file in given branch>>>>>>>
   ```
   这是预期行为；谁让文件不规范呢。

10. 更新完文件后，若分离点并非当前或给定分支头，则自动创建合并提交，日志消息为
    ```plaintext
    Merged [given branch name] into [current branch name].
    ```
    若存在冲突，额外在终端打印
    ```plaintext
    Encountered a merge conflict.
    ```
    合并提交（merge commit）记录两个父节点：当前分支头（第一父节点）与给定分支头。

视频演示见[此处](https://www.youtube.com/watch?v=JR3OYCMv9b4&t=929s)。

顺便说一句，提交历史已从线性序列 -> 树 -> 完整有向无环图（DAG）。

*   **运行时间**：O(N log N + D)，N 为两分支祖先提交总数，D 为所有文件数据量。

*   **失败场景**：  
  - 有暂存的增删，打印  
      ```plaintext
      You have uncommitted changes.
      ```
  
  - 分支不存在，打印  
      ```plaintext
      A branch with that name does not exist.
      ```
  
  - 合并自身，打印  
      ```plaintext
      Cannot merge a branch with itself.
      ```
  
  - 合并无变化，让普通提交错误提示通过即可。  
  - 未跟踪文件将被覆盖或删除，先打印  
      ```plaintext
      There is an untracked file in the way; delete it, or add and commit it first.
      ```
  
  并退出。

*   **危险？**：是！

*   **与真实 Git 差异**：  
  - 真实 Git 合并更细致，仅标记真正冲突区域。  
  - 分离点选择策略不同。  
  - 真实 Git 强制解决冲突后才提交；Gitlet 直接提交含冲突的版本，需额外提交修复。  
  - 真实 Git 会阻止对未暂存修改的文件进行合并；你可选择实现，但测试不覆盖。

*   **我们的代码行数**：约 70 行。

*   [**概念测验**](https://forms.gle/Gu4FcFf1kfC7HYBa6)

*   **建议先修**：Lecture 19（集合、映射、ADT）、Lecture 22（图遍历）

骨架代码
--------

骨架代码几乎空白，类里只有空方法与提示性 Javadoc。参考 Capers 项目：**主类**  
```plaintext
Main
```
  
本身不干活，只按  
```plaintext
args
```
  
分发调用。你可以删改其他类，但  
```plaintext
Main
```
  
必须保留，否则测试找不到入口。

若不知从何下手，先看 [Lab 6: Canine Capers](https://sp21.datastructur.es/materials/lab/lab6/lab6)。

设计文档
--------

由于本次没有完整的骨架代码，**我们要求每位同学提交一份设计文档，说明你的实现策略**。该文档不计分，但你在 Office Hours 或提交 Gitbug 求助前，必须提供一份最新且完整的设计文档。如果没有、未更新或不完整，我们将无法提供帮助。这是为了双方好：有了设计文档，你就拥有了一份完成任务的路线图。如果需要协助撰写设计文档，我们当然可以帮忙 :) 这里有[一些指导](https://sp21.datastructur.es/materials/proj/proj2/design.html)，以及一份 [Capers lab 的示例](https://sp21.datastructur.es/materials/proj/proj2/capers-example)。

评分细节
--------------

Gitlet 共有三个评分器：检查点评分器（checkpoint grader）、完整评分器（full grader）和快照评分器（snaps grader）。

> **学习批注：** checkpoint grader 只跑基础命令，确保你按时推进；full grader 跑全部测试用例；snaps grader 在后台随机抓取提交快照，用于防止期末突击。
### 检查点评分器

截止 3/12 23:59，可获 16 分额外加分。

提交到 Gradescope 上的 `Project 2: Gitlet Checkpoint`  自动评分器。

它将测试：

* 你的程序能编译通过。  
* 你通过骨架自带的样例测试：`testing/samples/*.in`。这些测试要求你实现：`init`、`add`、`commit`、`checkout -- [file name]`、`checkout [commit id] -- [file name]`、`log`。


此外，它会给出反馈（但不计分）：

* 是否通过风格检查（目前会忽略 `TODO`  类注释；最终提交时不会忽略）。

最终提交时，我们**会**对这些进行评分。3/4 更新：允许存在编译警告。

你最多拥有 1 个令牌，每 20 分钟刷新一次。失败时不会给出完整日志（只会告诉你哪个测试没过，无额外信息），不过既然你有测试本身，完全可以在本地调试。

> **学习批注：**  
> 令牌（token）机制防止频繁提交刷分，本地跑通 `make check` 再上 Gradescope 更稳。
### 完整评分器

截止 4/2 23:59，满分 1600 分。

完整评分器是一套更庞大、更全面的测试集。你最多拥有 1 枚令牌（token）。令牌补充节奏如下：

*   **2/20 - 3/19：** 每 6 小时 1 枚  
*   **3/20 - 3/26：** 每 3 小时 1 枚  
*   **3/26 - 4/2：** 每 20 分钟 1 枚  

和项目 1 一样，评分器访问次数受限。对自己好一点：边写代码边写测试，别把验证全丢给自动评分器。

与 checkpoint 相同，完整评分器会给出英文测试提示，但不会提供真正的 `.in`  文件。

> **学习批注：** 令牌机制迫使你本地先测，类比“省着用复活币”，每提交前确保本地测试通过，再花令牌验证。
### Snaps 评分器

截止日期：4 月 9 日晚 11:59。  
**只有在你把 snaps 仓库推送并提交到 Snaps Gradescope 作业后，Gradescope 的分数才会同步到 Beacon。**  
推送 snaps 仓库的命令如下：

```
cd $SNAPS_DIR
git push
```


推送完成后，在 Gradescope 提交你的 `snaps-sp21-s***` 仓库（与项目 1 类似）。  
> **学习批注：** 此提交仅针对完整评分器，不包含 checkpoint 与额外加分作业。

若忘记推送，可在截止后一周内补交；超过一周则需使用 slip days。
### 额外加分（Extra credit）

共有 16 + 32 + 64 = 112 分额外加分：

1.  16 分给检查点  
2.  32 分给 `status`  命令打印
    ```plaintext
    Modifications Not Staged For
    Commit
    ```
 与 `Untracked Files`  部分  
3.  64 分给远程命令  

其余规范已为你准备好，务必阅读。**测试与调试部分将极其有用**，因为本项目的测试方式与之前不同，但并不复杂。

---
### 项目须知杂项

呼！刚才一口气讲了太多命令。别担心，难度并不相同。每个命令旁标注了我们实现时的大致代码行数（仅统计该命令独有代码，复用部分不计）。无需完全对齐，但能帮你估算各命令耗时。合并（merge）比其他命令长，别拖到最后！

本项目颇有挑战，刚开始摸不着头脑很正常。因此，可适度加强合作，但注意：

*   在 `gitlet/Main.java`  文件开头注释里列出所有协作者。  
*   不分享具体代码；每位同学必须独立实现算法，以便我们看出差异。

Ed 综合讨论串通常极长，却充满高质量思路。积极利用班级规模，搜一搜是否已有类似疑问（若与设计相关的独有 bug，请提交 Gitbug）。

---
### 文件操作提示

项目需读写文件。 `java.io.File`  和 `java.nio.file.Files`  类可能派上用场。其实 `java.io`  与 `java.nio`  包里也有不少好东西。务必浏览 ` `gitlet.Utils`  包`，我们已写好部分工具。深入挖掘，或许能发现让 IO 部分轻松数倍的方法！警告：若你开始用 Reader、Writer、Scanner 或 Stream，说明把问题复杂化了。

> **学习批注：** 优先用 `Utils.writeContents` / `readContents` 这类封装，少碰底层流。

---
### 序列化（Serialization）细节

Gitlet 每次只能执行一条命令，因此必须跨命令记住提交树。这意味着：不仅要设计运行时类表示内部结构，还要在 `.gitlet` 目录里保存对应文件，供下次运行读取。

简便做法：把需持久化的运行时对象直接序列化。Java 会自动搞定字段到字节的转换。

lab6 已学过序列化，此处不重复。若仍困惑，回看 lab6 相关部分。

注意陷阱：Java 序列化会“追指针”。即，传入 `writeObject`  的对象及其引用链都会被一并写出。若提交对象里用指针指向父提交，写分支头时会把整条提交子图（含 blobs）全塞进一个文件，通常不是你想要的。

解决：运行时用 SHA-1 哈希字符串而非 Java 指针来引用提交与 blob；维护“哈希→对象”的内存映射，启动时填充，永不存盘。

为省查找时间，可在对象里同时保留（冗余）指针与哈希字符串，并把指针字段标为 `transient`：

```
private transient MyCommitType parent1;
```


`transient` 字段不会被序列化；反序列化后值为 null，需手动恢复。

> **背景扩展：** 序列化文件是二进制，用文本编辑器打开一片乱码。我们提供调试小工具 `gitlet.DumpObj` ，详见 `gitlet/DumpObj.java`  的 Javadoc。

---
### 测试

务必通读本节，也可观看[视频](https://www.youtube.com/watch?v=uMYpuQuHGu0&t=752s)。

测试算分。为每条命令写集成测试，覆盖全部功能；单元测试随意。我们不提供单元测试，因其与实现紧密相关。

已提供集成测试框架 `testing/tester.py` ，解析 `.in` 文件。运行全部测试：

```
make check
```


想看失败详情（含你的输出）：

```
make check TESTER_FLAGS="--verbose"
```


单测：在 `testing`  子目录执行

```
python3 tester.py --verbose FILE.in ...
```


其中 `FILE.in ...`  为指定的 `.in`  文件列表。

**注意**：运行测试前必须手动 `make`  重新编译！

命令

```
python3 tester.py --verbose --keep FILE.in
```


会在出错时保留 `tester.py`  生成的目录，方便你检查文件状态；测试通过也会保留最终内容。

测试器实现了一套极简**领域专用语言（DSL）**，支持常用断言与命令。

*   在测试目录里增删文件；  
*   运行 `java gitlet.Main` ；  
*   比对 Gitlet 输出与期望字符串或正则表达式；  
*   检查文件是否存在、缺失及内容。  
执行
```plaintext
python3 testing/tester.py
```
  
（无参数，如上所示）可查看该测试语言的说明。我们在  
```plaintext
testing/samples
```
  
目录里放了一些示例。不要把你的测试放在该子目录，另建独立文件夹，以免与官方测试（可能含 bug）混淆。把所有  
```plaintext
.in
```
  
文件放到  
```plaintext
testing
```
  
目录下的  
```plaintext
student_tests
```
  
文件夹中。骨架里该文件夹初始为空。

> **学习批注：** 官方测试与你自己的测试分开放，调试时一眼就能定位是谁的锅。

我们给 Makefile 加了兼容处理：若你系统里 Python 3 的命令就是 `python` ，可直接用我们的 makefile，无需改动：

```
make PYTHON=python check
```


还能给  
```plaintext
tester.py
```
  
追加参数，例如：

```
make TESTER_FLAGS="--keep --verbose"
```


使用 Staff 方案测试
-----------------------------

截至 2 月 28 日（周日），你可以用 staff solution 验证命令理解及自写测试，指南见[这里](https://sp21.datastructur.es/materials/guides/staff-gitlet)。

理解集成测试
-------------------------------

无论是 Gitbugs 提 issue，还是去 Office Hours 求助，我们第一句话都是：“把你挂掉的测试给我看看”。因此，学会写测试是本项目核心技能。我们已尽量让这一步无痛，请务必读完本节，看懂官方测试并写出高质量自测。

集成测试格式与 Capers 类似。若你不知道 Capers 的  
```plaintext
.in
```
  
文件怎么跑，先去读 [capers 规范](https://sp21.datastructur.es/materials/lab/lab6/lab6) 对应章节。

官方测试远不够全面，想拿满分必须自己补测。写测试前，先搞清整体机制。

```plaintext
testing
```
  
目录结构如下：

```
.
├── Makefile
├── student_tests                    <==== Your .in files will go here
├── samples                          <==== Sample .in files we provide
│   ├── test01-init.in               <==== An example test
│   ├── test02-basic-checkout.in
│   ├── test03-basic-log.in
│   ├── test04-prev-checkout.in
│   └── definitions.inc
├── src                              <==== Contains files used for testing
│   ├── notwug.txt
│   └── wug.txt
├── runner.py                        <==== Script to help debug your program
└── tester.py                        <==== Script that tests your program
```


与 Capers 一样，测试会在  
```plaintext
testing
```
  
里建临时目录，然后按  
```plaintext
.in
```
  
文件跑命令。若加  
```plaintext
--keep
```
  
标志，临时目录会保留，方便你事后翻现场。

与 Capers 不同，我们得检查工作目录（working directory）里文件的**内容**。因此  
```plaintext
testing
```
  
下多了  
```plaintext
src
```
  
文件夹，里面预置了大量  
```plaintext
.txt
```
  
文件，内容特定。后文再细讲，现在只需记住：  
```plaintext
src
```
  
存真实文件内容；  
```plaintext
samples
```
  
放样例测试（即 checkpoint 测试）的  
```plaintext
.in
```
  
文件。自建测试请放到骨架里初始为空的  
```plaintext
student_tests
```
  
文件夹。

Gitlet 的  
```plaintext
.in
```
  
文件支持更多函数，直接看  
```plaintext
tester.py
```
  
里的说明：

```
# ...  A comment, producing no effect.
I FILE Include.  Replace this statement with the contents of FILE,
      interpreted relative to the directory containing the .in file.
C DIR  Create, if necessary, and switch to a subdirectory named DIR under
      the main directory for this test.  If DIR is missing, changes
      back to the default directory.  This command is principally
      intended to let you set up remote repositories.
T N    Set the timeout for gitlet commands in the rest of this test to N
      seconds.
+ NAME F
      Copy the contents of src/F into a file named NAME.
- NAME
      Delete the file named NAME.
> COMMAND OPERANDS
LINE1
LINE2
...
      Run gitlet.Main with COMMAND ARGUMENTS as its parameters.  Compare
      its output with LINE1, LINE2, etc., reporting an error if there is
      "sufficient" discrepency.  The <<< delimiter may be followed by
      an asterisk (*), in which case, the preceding lines are treated as
      Python regular expressions and matched accordingly. The directory
      or JAR file containing the gitlet.Main program is assumed to be
      in directory DIR specifed by --progdir (default is ..).
= NAME F
      Check that the file named NAME is identical to src/F, and report an
      error if not.
* NAME
      Check that the file NAME does not exist, and report an error if it
      does.
E NAME
      Check that file or directory NAME exists, and report an error if it
      does not.
D VAR "VALUE"
      Defines the variable VAR to have the literal value VALUE.  VALUE is
      taken to be a raw Python string (as in r"VALUE").  Substitutions are
      first applied to VALUE.
```


别担心上面提到的 Python 正则：其实超简单，稍后举例秒懂。

走读一个完整测试，看看从启动到收尾都发生了什么。我们来拆 `test02-basic-checkout.in` 。
#### 示例测试

首次运行此测试时，会创建一个空临时目录，目录结构如下：

```
.
├── Makefile
├── student_tests
├── samples
│   ├── test01-init.in
│   ├── test02-basic-checkout.in
│   ├── test03-basic-log.in
│   ├── test04-prev-checkout.in
│   └── definitions.inc
├── src
│   ├── notwug.txt
│   └── wug.txt
├── test02-basic-checkout_0          <==== Just created
├── runner.py
└── tester.py
```


该临时目录即本次测试专用的 Gitlet 仓库（Gitlet repository），所有操作都在此执行。若再次运行测试而未删除旧目录，系统会新建名为 `test02-basic-checkout_1`  的新目录，依此类推。每次测试独占一个目录，互不干扰，无需担心冲突。

测试首行为注释，直接忽略。

下一段：

```
> init
```


此段无输出（首行 `>` 与代码围栏结束符之间无文本），但会生成 `.gitlet` 文件夹。此时目录结构：

```
.
├── Makefile
├── student_tests
├── samples
│   ├── test01-init.in
│   ├── test02-basic-checkout.in
│   ├── test03-basic-log.in
│   ├── test04-prev-checkout.in
│   └── definitions.inc
├── src
│   ├── notwug.txt
│   └── wug.txt
├── test02-basic-checkout_0
│   └── .gitlet                     <==== Just created
├── runner.py
└── tester.py
```


继续：

```
+ wug.txt wug.txt
```

该行使用 `+` 命令，将 `src` 目录右侧文件内容复制到临时目录左侧文件（若不存在则新建）。两文件同名无妨，因路径不同。执行后目录结构：

```
.
├── Makefile
├── student_tests
├── samples
│   ├── test01-init.in
│   ├── test02-basic-checkout.in
│   ├── test03-basic-log.in
│   ├── test04-prev-checkout.in
│   └── definitions.inc
├── src
│   ├── notwug.txt
│   └── wug.txt
├── test02-basic-checkout_0
│   ├── .gitlet
│   └── wug.txt                     <==== Just created
├── runner.py
└── tester.py
```


可见 `src`  目录用途：存放测试所需文件内容，按需置入仓库。若想给文件写入特殊内容，先在 `src`  建同名文件，再用 `+`  命令复制。注意参数顺序：右侧为 `src`  目录文件，左侧为临时目录文件，别搞反。

下一段：

```
> add wug.txt
```


无输出。此时 `wug.txt`  文件已暂存（staged）待添加。你的 `test02-basic-checkout_0/.gitlet`  目录结构可能随之变化，需持久化记录 `wug.txt`  的暂存状态。

再下一段：

```
> commit "added wug"
```


仍无输出， `.gitlet`  内部结构或再变。

接着：

```
+ wug.txt notwug.txt
```

因临时目录已存在 `wug.txt`，其内容被替换为 `src/notwug.txt` 的版本。

下一节：

```
> checkout -- wug.txt
```


依旧无输出，但会把临时目录中 `wug.txt`  内容恢复为 `src/wug.txt`  的原版。随后用断言验证：

```
= wug.txt wug.txt
```


此为断言：若左侧临时目录文件内容与右侧 `src`  目录文件不完全一致，测试脚本报错并提示文件内容不符。

另有两种断言命令：

```
E NAME
```


断言临时目录存在名为 `NAME`  的文件或文件夹（仅检查存在性，不验内容）。不存在则测试失败。

```
* NAME
```

断言临时目录**不存在**名为 `NAME` 的文件或文件夹。若存在则测试失败。

测试到此结束。若带 `--keep` 标志，临时目录保留，否则自动删除。若怀疑 `.gitlet` 目录初始化或持久化有问题，可保留目录手动排查。

> **学习批注：**
> 测试框架通过“复制-断言”模式工作：先用 `cp` 把 `testing/src` 里的标准文件搬进临时仓库，再跑 Gitlet 命令，最后用 `assert` 比对结果。类比做菜：`testing/src` 是备料区，临时目录是灶台，断言就是尝味道。
#### 测试准备

你很快会发现，测试某个命令常常需要重复准备：比如要测试 `checkout` 命令，你得：

1. 初始化一个 Gitlet 仓库  
2. 创建一次 `提交（commit）`，把文件存成某个版本（v1）  
3. 再创建一次 `提交`，把同文件改成另一版本（v2）  
4. 把该文件 `checkout` 回 v1  

如果想测“第二次提交未跟踪但第一次提交跟踪”的文件，步骤还会更多。

省时的办法是把这套准备写进一个文件，然后用 `I` 命令加载。示例如下：

```
# Initialize, add, and commit a file.
> init
+ a.txt wug.txt
> add a.txt
> commit "a is a wug"
```


把它放到 `samples`  目录，扩展名用 `.inc` ，比如叫 `samples/commit_setup.inc` 。若误用 `.in`  扩展名，测试脚本会把它当成独立测试单独跑。正式测试里只需一句：

```
I commit_setup.inc
```


脚本会执行文件里所有命令，并保留创建的临时目录，测试脚本因此更短、更易读。

我们已备好一份 `.inc`  文件，名为 `definitions.inc` ，帮你快速搭建“模式（patterns）”。接下来看看模式是什么。

> **学习批注：**  
> 把重复 setup 抽成“小剧本”文件，就像拍短片前先写好分镜，主演（真正测试）上场时直接喊“Action”，省时又清晰。
#### 模式匹配输出

测试里最绕的部分就是像 `log`  这样的输出，原因有三：

1. 提交（commit）SHA 会随代码改动不断变化，你得不停改测试。  
2. 时间永远向前，日期字段每次都会变。  
3. 这让测试长得离谱。

我们其实并不关心“一字不差”，只要有个 SHA、日期格式对即可。因此测试用**模式匹配（pattern matching）**。

> **学习批注：** 想象给答案留空：只检查“这里填的是学号”，而不在乎具体数字。

你不需要深究其原理，只要知道：我们给文本定个“样子”（如提交 SHA），然后验证输出里出现该样子即可。

下面演示如何对 `log`  的输出做模式匹配：

```
# First "import" the pattern defintions from our setup
I definitions.inc
# You would add your lines here that create commits with the
# specified messages. We'll omit this for this example.
> log
===
${COMMIT_HEAD}
added wug

===
${COMMIT_HEAD}
initial commit

<<<*
```


这段跟普通 Gitlet 命令一样，只是结尾多了 `<<<*` ，告诉测试脚本启用模式。模式被包在 `${PATTERN_NAME}`  里。

所有模式定义在 `samples/definitions.inc` 。你只需知道它“长什么样”，不必读懂正则。例如 `HEADER`  匹配的是提交头，大概这样：

```
commit fc26c386f550fc17a0d4d359d70bae33c47c54b9
```


那就是一段随机提交 SHA。

所以写期望输出时，只要知道 log 有几条、提交信息是什么即可。

对 `status`  命令也能这么干：

```
I definitions.inc
# Add commands here to setup the status. We'll omit them here.
> status
=== Branches ===
\*master

=== Staged Files ===
g.txt

=== Removed Files ===

=== Modifications Not Staged For Commit ===

=== Untracked Files ===
${ARBLINES}

<<<*
```


这里用的模式是 `ARBLINES` ，代表“任意多行”。若你真关心 untracked 文件，可去掉模式直接写；但通常我们更想看 `g.txt`  已暂存待添加的文件。

注意分支上的 `\*` 。回忆 `status` 命令里，当前分支（HEAD）前要加 `*` 。若用模式，期望输出里得把 `*` 换成 `\*`，这叫“转义”星号。若不用模式（命令以 `<<<` 而非 `<<<*` 结尾），可直接写 `*`，无需 `\` 。

模式还能“保存”匹配到的片段。**警告**：这像魔法，你完全不必懂原理，会用即可。可直接抄我们提供的测试，不用手搓。

若执行 `checkout`  命令，就得用 SHA 指定要检出的提交。但我们用了模式，事先不知道 SHA。解决办法是用 `test04-prev-checkout.in`  把 SHA“抓”出来：

```
I definitions.inc
# Each ${COMMIT_HEAD} captures its commit UID.
# Not shown here, but the test sets up the log by making many commits
# with specific messages.
> log
===
${COMMIT_HEAD}
version 2 of wug.txt

===
${COMMIT_HEAD}
version 1 of wug.txt

===
${COMMIT_HEAD}
initial commit

<<<*
```


这条命令会在 `log`  后把 UID（SHA）捕获下来。接着用 `D`  把 UID 存进变量：

```
# UID of second version
D UID2 "${1}"
# UID of first version
D UID1 "${2}"
```


注意编号是倒序：从 log 最上面开始，从 1 计数。因此当前版本（第二条）被定义为 `"${1}"` 。初始提交我们不关心，就不抓它的 UID。

现在就能用存好的 SHA 做检出：

```
> checkout ${UID1} -- wug.txt
```


然后写断言，确认检出成功即可。
#### 测试结论

测试脚本能做的复杂事还有很多，但掌握这些就足够写出非常棒的测试。  
把官方提供的测试当范例起步，也欢迎在 Ed 上交流测试思路（不要贴完整代码）。  
可分享 `.in`  文件，但务必先确保正确，并加注释方便同学和助教看懂。

调试集成测试
---------------------------

回忆 [Lab 6](https://sp21.datastructur.es/materials/lab/lab6/lab6)：在新框架下调试集成测试略有不同。 `runner.py`  脚本与 Capers 阶段用法一致，先回顾 Lab 6 对应章节并观看配套视频。  
下面列出调试策略：

### 定位真正出错的执行

每个测试会多次运行你的程序，每一次都可能引入 bug。首要任务是**找出哪一次执行导致了问题**。换句话说：假设你有一个测试在检查 `status`  命令时失败，输出只差一个文件——你说它未跟踪（untracked），但测试认为它已暂存待添加（staged for addition）。**这并不一定意味着 `status`  命令有 bug**。可能是 `status`  命令写错了，但也可能是 `add`  命令**没有把“文件已暂存”这一信息持久化**！如果这样，即使 `status`  命令完全正确，程序也会报错。

> **学习批注：** 把 bug 想象成“多米诺骨牌”，第一张牌倒在哪一步，后面全错。先找到第一张牌，再修最后一张没意义。

如何定位？用 `runner.py`  脚本一步步跑，每跑完一次就检查临时目录，确认文件是否写对。对于**序列化（serialization）**对象，内容是一堆看不懂的字节流，你只需在序列化那一刻确认对象字段正确即可；有时你会发现**根本没序列化**！

实在找不到再来 Office Hours 或发 Gitbug。注意：Office Hours 每人限 10 分钟，复杂 bug 直接发 Gitbug，并附上**尽可能全的信息**。别忘了更新设计文档，否则 Gitbug 会被拒。

远程命令（额外加分）
---------------------------

本项目只模拟 git 的本地功能，真正的威力在**远程（remote）**协作。你可以把改动推给队友，反之亦然，共享完整历史。

想拿 64 分附加分？实现以下基础远程命令： `add-remote` 、 `rm-remote` 、 `push` 、 `fetch` 、 `pull` 。**先完成主项目再碰加分**，否则性价比极低。我们优先保证大家通关主线；做加分需更独立。

命令说明
------------

*   执行时间不计分，但别写离谱算法。  
*   所有命令都已大幅简化，与真实 git 的差异默认不标，请自行留意。
#### add-remote

*   **用法**：  
    ```plaintext
    java gitlet.Main add-remote [remote name] [name of remote directory]/.gitlet
    ```


*   **描述**：将给定的登录信息保存在指定的远程名称（remote name）下。后续对该远程名称执行 push 或 pull 时，程序会尝试使用这个 `.gitlet`  目录。例如，执行 `java gitlet.Main add-remote other ../testing/otherdir/.gitlet`，就能在任意路径（本地或 grading 程序内）测试远程仓库。命令中务必全部使用正斜杠 `/`；程序内部再把 `/` 转换成当前系统的路径分隔符（Unix 为 `/`，Windows 为 `\`）。Java 贴心地提供了类变量 `java.io.File.separator`  来表示该分隔符。

> **学习批注：** 路径字符串写死为 `/` 可保证跨平台脚本一致，类似“写 SQL 都用单引号”的约定。

*   **失败情形**：若已存在同名远程，打印错误信息：  
    ```plaintext
    A remote with that name already exists.
    ```
  
无需验证用户名或服务器是否真实存在。

*   **危险操作？**：否。
#### rm-remote

*   **用法**：  
    ```plaintext
    java gitlet.Main rm-remote [remote name]
    ```


*   **描述**：删除给定远程名称（remote name）的相关信息。如果你想更换已添加的远程仓库，必须先移除再重新添加。

*   **失败情况**：若该名称的远程不存在，打印错误信息：  
    ```plaintext
    A remote with
    that name does not exist.
    ```


*   **危险操作？**：否。

> **学习批注：** 类似 `git remote remove origin`，但 Gitlet 只维护本地 `.gitlet/remotes/` 下的文本文件，删除即删掉对应文件，不会触碰真正的远程仓库。
#### push

*   **用法**：  
    ```plaintext
    java gitlet.Main push [remote name] [remote branch name]
    ```

*   **描述**：尝试将当前分支（branch）的提交追加到指定远程仓库的对应分支末尾。细节如下：

  仅当远程分支的头指针（HEAD）在当前本地头指针的历史中存在时，命令才会生效；换句话说，本地分支拥有远程分支“未来”的提交。此时把这些“未来”提交追加到远程分支，然后远程仓库重置到追加后的最前端（使其头指针与本地一致）。这一过程称为快进（fast-forward）。

  若远程机器上的 Gitlet 已存在但缺少该分支，则直接为其新增该分支。

*   **失败情形**：  
  若远程分支的头指针不在当前本地头指针的历史中，打印错误信息
  ```plaintext
  Please pull down
  remote changes before pushing.
  ```

  若远程目录 `.gitlet` 不存在，打印
  ```plaintext
  Remote directory not found.
  ```


*   **危险？**：否。

> **学习批注：** 想象两队接力跑，本地队已多跑了一圈；push 就是把本地多跑的那圈“快进”给远程队，只要远程队当前位置在本地队历史轨迹上即可。
#### fetch

*   **用法**：  
    ```plaintext
    java gitlet.Main fetch [remote name] [remote branch name]
    ```

*   **描述**：把远程 Gitlet 仓库的提交（commit）拉取到本地 Gitlet 仓库。本质上，它会复制远程仓库指定分支中**尚未存在于本地**的所有提交和 blob，并在本地 `.gitlet`  中创建一个名为 `[remote name]/[remote branch name]`  的分支（branch）（就像真正的 Git 一样），然后把 `[remote name]/[remote branch name]`  指向该分支的最新提交，从而把远程分支的内容同步过来。如果本地尚不存在该分支，会自动创建。

> **学习批注：** 想象你在书店借书：远程仓库是总店，本地是分店。fetch 相当于把总店的新书目录复印一份放分店，但你还不能把书带回家（工作目录不变），只是让分店有了这些书的信息。

*   **失败场景**：  
  - 若远程仓库没有指定分支名，打印错误
      ```plaintext
      That remote does not have that
      branch.
      ```
  
  - 若远程 `.gitlet`  目录不存在，打印
      ```plaintext
      Remote directory not found.
      ```


*   **危险？** 否

#### pull

*   **用法**：  
    ```plaintext
    java gitlet.Main pull [remote name] [remote branch name]
    ```

*   **描述**：抓取分支 `[remote name]/[remote branch name]` ，就像执行 `fetch`  命令一样，随后将抓取到的内容合并（merge）到当前分支。

*   **失败场景**：同时触发 `fetch`  和 `merge`  的所有失败情况。

*   **危险？** 是的！

I. 需要避免的做法
------------------

经验表明，以下做法会让你陷入无尽的痛苦：程序跑不通、难以复现的 bug（“海森堡 bug”）。

1.  你可能会把各种信息（如提交）存进文件，然后顺手用文件系统操作（如列目录）遍历它们。小心！ `File.list`  和 `File.listFiles`  返回的文件名顺序是不确定的。如果你靠它们来实现 `log`  命令，结果会随机变化。

2.  Windows 用户特别注意：Unix（或 macOS）的路径分隔符是 `/` ，而 Windows 是 `\`。如果你在代码里手动拼接目录和文件名时用了硬编码的 `/`  或 `\` ，换系统肯定翻车。Java 提供了系统相关的分隔符（ `System.getProperty("file.separator")` ），或者直接用 `File`  的多参数构造函数。

3.  做序列化（serialization）时慎用 `HashMap` ！其内部顺序不确定，会导致反序列化后对象状态不一致。解决方法是改用 `TreeMap` ，顺序永远固定。更多细节见[这里](https://stackoverflow.com/questions/5993752/hashmap-serialization-and-deserialization-changes)。

> **学习批注：** 用 `HashMap` 序列化后，下次加载遍历顺序可能变，日志或测试就会“随机”失败；换成 `TreeMap` 就能根治。

J. 致谢
------------------

感谢 Alicia Luengo、Josh Hug、Sarah Kim、Austin Chen、Andrew Huang、Yan Zhao、Matthew Chow，特别是 Alan Yao、Daniel Nguyen 和 Armani Ferrante 对本项目的反馈。感谢 git 带来的极致体验。

本项目深受 Philip Nilsson 的[这篇][Nilsson Article]优秀文章启发。

项目最初由 Joseph Moghadam 创建；2015 秋、2017 秋、2019 秋由 Paul Hilfinger 维护更新。
