## Abstract

## Introduction
在面向AGI的时代，基于大语言模型的智能体（LLM agents）逐渐从封闭的文本生成系统走向能够自主调用工具、访问外部信息并执行多步任务的交互式系统。通过调用搜索引擎、邮件服务、数据库等接口和工具，智能体能够与日益开放复杂的环境进行交互并处理用户请求。然而，卓越的工具调用能力也拓展了安全风险：智能体在执行用户任务时，往往需要读取来自网页、邮件、文档、数据库或第三方 API 的外部内容，不怀好意的攻击者可以借此操控外部环境，将恶意指令嵌入工具返回值中，并利用智能体对外部信息的依赖，诱导其执行与原始任务无关甚至具有破坏性的工具调用。鉴于这些攻击（间接提示注入，IPI）不需要修改系统提示词、用户指令或模型参数，易于实施和扩展，该风险已成为智能体安全攻防双方拉锯的主要战场。




引出两个insight：
知己知彼，方能百战不殆。从攻击器的角度思考，目前的间接注入攻击太过直接，意图过于明显（引入delta里面的一个例子）。agent在正常执行酒店查询任务中，send_email的注入显得生硬缺乏伪装。尽管现在已有一些工作尝试使用自适应的办法挖掘域外的间接提示注入攻击，但其本质依赖大语言模型的世界知识进行发散，并没有提供一个确切的优化目标。
对于好的攻击器，核心思路是消除间接注入带来的行为突变，将恶意意图伪装成正常业务流程的一部分。由此引出优化目标。




从防御器的角度思考，好的防御器不应只局限于对静态间接提示注入攻击有抵抗力，它应该不断进化，从而对各种域外的自适应攻击均具备抵抗能力。
对于好的防御器，核心思路是和攻击器一起协同进化（点一下优化目标-delta），在增强防御能力的同时，同时保证良性任务的效用能力。

## related work
目前打算分两类：
evolution in agent safety和agent IPI defense
**evolution in agent safety**:
这里可以做一个简单的分类，比如memory进化的Safin-1: Safety from Within through Memory-Native State Evolution
训练模型进化的MAGIC GPT-red
skill进化的skillaudit
再补点其他ref上去

**agent IPI defense**:
training based defenses安全对齐类的: secalign,meta-secalign,secOPD,reasalign,localalign,struQ，COPA，RETA……
Filtering-based defenses护栏模型、分类模型类的：shieldagent，clawguard，TS-Guard（toolsafe）
其他的（xxxx 这里可以自行总结一个粒度和前面对齐的类别）：promptarmor，MELON……

## Preliminary
首先分析，在间接提示注入攻击中，干净任务，攻击成功任务，攻击失败任务之间有什么差别？——preliminary文件下的实验（agentdojo数据集，seq_entropy和seq_nll在late段表现出来的特征是AF>AS>clean，这表明模型在突然看到工具返回值中出现明显偏离原始正常任务的指令时，会表现出犹豫与警觉）。所以，消除间接注入带来的行为突变，将恶意意图伪装成正常业务流程的一部分，是攻击器的一个优化目标。

接着，我们在agentdojo原始数据集上，人工construct了几十条新的升级版攻击（delta.md里面的是一个case），发现在该子集上，现有的几个防御基线asr均显著的高。（protectAI，PIGuard，meta-secalign，base）
由此formulate一个攻击器优化目标：Δ 从间接提示被注入开始，到智能体开始发生行为转变，间隔的轮次。轮次越大，表明攻击隐蔽性越好，攻击效果也就越好。