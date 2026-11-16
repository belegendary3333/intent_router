# 本地意图分析台

分析台按 chat / skill / singing / dancing / focus 五类意图比较路由方案。`routes.yaml` 包含 88 条基础样例，独立分析语料共 750 条；分析语料为合成数据，不代表生产真实数据。

## 启动

在克隆后的项目根目录执行以下命令。默认端口 **8892**，与默认使用 8890 的路由服务独立。

```bash
bash run_lab.sh
```

打开 [本地分析台](http://127.0.0.1:8892/)。环境已安装、模型已准备的情况下，不需要再次下载。页面会显示 small / large 分别分析的进度；首次 large 分析需要重建向量，耗时取决于硬件。模型准备失败时会显示原因。

首次安装或在新机器上使用：

```bash
bash setup_lab.sh
bash run_lab.sh
```

使用独立 `.venv-lab`，默认 Python 3.13；可指定 `LAB_PYTHON=3.12 bash setup_lab.sh`。当前锁定的 NumPy/SciPy 要求 Python 3.12 及以上。依赖安装到 `.venv-lab` 中。安装需要网络；模型和依赖准备好后，网页、图表和推理均可离线使用，没有 CDN 请求。

如果 8892 被占用，脚本会明确提示，不会停止其他进程。可打开已运行的页面，或另选端口：

```bash
LAB_PORT=8893 bash run_lab.sh
```

只监听 `127.0.0.1`。终端前台启动时，`Ctrl+C` 停止。分析台独立运行，候选结果保存在本地输出目录，不会自动写回 `routes.yaml` 或部署到路由服务。

## small / large 对照

支持下载 `BAAI/bge-large-zh-v1.5` 的公开 INT8 ONNX 转换，模型文件约 **327.4 MB**，保存在 `model/large/`，与 `model/` 中的 small 分开保存。模型权重不包含在 Git 仓库中。

在新机器上补充 large：

```bash
bash setup_lab.sh
bash setup_large_lab.sh
bash run_lab.sh
```

只有 small 时仍可使用分析台；large 下载完成后重启分析台，自动构建双模型。可用 `LAB_LARGE_MODEL_DIR=/绝对路径` 指定 large 目录，不能用 `LAB_MODEL_DIR` 指向 large；后者仍是 small 配置。

- 两边共用 **同一个 `routes.yaml` 的 88 条模式设定词**、同一个 `analysis_lab/cases.yaml`、相同数据划分、KMeans seed=42 / n_init=10 / 每类锚点数、CLS pooling / L2 归一化 / CPU / 4 线程 / 512-token 截断。报告校验数据和设定词一致后才生成。
- 两边均是 INT8，但 small 的转换来自 `onnx-community`，large 来自 `Xenova`，量化细节可能不同；这比较的是两套实际部署版本，不能等同于原始 FP32 模型的公平能力排名。
- 右侧并排画 small / large；范围、perplexity 和 seed 共用。两图各自 PCA+t-SNE，二维轴的方向、尺度和坐标不能直接相减或据此排名。
- 发送一句话会同步测试两边；金色星形参加各自真正的 t-SNE 重算。推理失败会回滚两边的新探针，避免记录错位。
- “模型对比”页显示原始向量空间五类标签的余弦轮廓系数、类内平均余弦、KMeans 聚类一致度 ARI、代表锚点相同数、完整测试指标、预测不同的测试句，以及共同设定词。
- “详情模型”选择 small 或 large 后，锚点、混淆句、测试报告、探针与下载对应所选模型。候选锚点各自从训练集选择，可能不同；原有设定词始终不变。
- 阈值调参继续只用验证集，两模型分别调参；默认阈值对照和调参后结果同时保留。没有自动替换原有路由模型、写回阈值、commit 或 push。
- 单句耗时基准对同样 12 条测试短句，预热 3 次、每句测 2 次，包含 tokenizer+ONNX+归一化；不含网页、t-SNE、路由和网络。短小样本 P50/P95 只用于同机对照，不是生产延迟保证。

large 转换来源：[Xenova/bge-large-zh-v1.5](https://huggingface.co/Xenova/bge-large-zh-v1.5)，基础模型：[BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)。下载使用公开 ModelScope 镜像固定 revision `88c72f156f26f7ae08bd4c8bd1b5ffaf668240c5`，三个文件均校验固定 SHA256，来源记录在 `model/large/provenance.json`。校验不符不会覆盖或加载。

测试中聊天分散并不意味着必须替换模型。请优先看真实标注对话的“聊天误触发动作”和每类召回率；750 条合成句的分数不能代表线上准确率，也不要反复按测试集误判修改测试标签。

## 页面怎么用

1. 左侧发送原句，每行一句，单句最多 256 字符，一次最多 20 句。
2. 可选择期望意图，用于人工标注探针统计；留空时只显示推断结果，不计算准确率。
3. 左侧显示同一句在两模型中的扩充样例结果，以及所选详情模型的原有样例、扩充样例、锚点三套方案。所有新句使用原有阈值 `0.62 / 0.03`，不会自动应用候选阈值。
4. 新句以金色星形加入散点图。悬停查看原句，点击查看意图、集合和表达家族；支持缩放、拖动和恢复视图。
5. 选择五类联合图或单类内部图；对比 perplexity 20/30/50、随机种子 42/7。
6. 下方查看锚点、易混淆句、混淆矩阵、每类 Precision/Recall/F1、测试集误判和探针统计。每类锚点可选 10/15/20，点击“重新分析”。

输入框分析整条输入的意图，不生成聊天回复、不驱动机器人、不处理跨轮指代或上下文。新句不会自动写入样例库；最近 100 条探针只保存在当前进程内存，重启后消失。下载的报告包含当前探针与人工标注，分享前请注意个人信息。

## 案例与隔离

`analysis_lab/cases.yaml` 是可编辑的 750 条案例配置：聊天、任务、唱歌、跳舞、专注模式各 150 条。

| 集合 | 每类 | 总数 | 用途 |
| --- | ---: | ---: | --- |
| train | 100 | 500 | 扩充样例库、选锚点 |
| validation | 20 | 100 | 选择候选分数/差值阈值 |
| test | 30 | 150 | 独立比较方案 |

每条有 `id`、`text`、`mode`、`split`、`family`、`subtype`、`source`。`family` 是表达家族，不可跨集合。`corpus.py` 保留首版人工编写的生成源，不是简单的同义词笛卡尔积。修改 `cases.yaml` 后点击“重新分析”，或重启网页服务。

校验规则：五类各 100–200 条，每个集合至少 10 条；ID 和去除唤醒词后的文本不能重复；验证/测试集不能与原有 `routes.yaml` 精确重复。报告另外告警其与训练/原有样例余弦相似度 ≥ 0.97 的语义近重复。没有告警也不能证明不存在泄漏或所有标签正确。

案例是**人工编写的合成数据，不是真实对话日志**；包含委婉请求、陈述、讨论和否定句等边界场景。首次仅作探索，后续应加入人工核验的真实 ASR 语句，并重新划分独立验证/测试集，不要按测试结果反复修改它直到分数变高。

## 模型、t-SNE 和锚点

- 模型：small 为 `BAAI/bge-small-zh-v1.5` / 512 维，large 为 `BAAI/bge-large-zh-v1.5` / 1024 维；均直接 ONNX Runtime CPU 推理、CLS pooling、L2 归一化。small 保留与独立路由服务的同环境数值一致性测试。
- 文件：公开的 `onnx-community/bge-small-zh-v1.5-ONNX` INT8 文件及 tokenizer，沿用此前核验的 SHA256；下载器使用 ModelScope 镜像，若现有文件哈希不符会拒绝覆盖。可用 `LAB_MODEL_DIR=/绝对路径` 指定另一份已核验模型目录。
- 动态 INT8 的结果会受批次组成/填充长度影响。因此每条案例/查询单独编码；分类参考库则与原有服务一样整体编码。基线与原有服务在同一运行环境中有数值一致性测试；跨 CPU/Runtime 版本仍可能有微小差异。
- 可视化：各自原始 512 / 1024 维 → PCA 50 维 → t-SNE 2 维。五类联合图用于观察跨类邻域，单类图用于探索内部子簇。邻域保持度不是分类准确率。
- 新句参加真实 t-SNE 联合重算，不是近邻坐标插值。加入新句会使原有点移动；不同图的二维坐标不能直接比较。
- 锚点：只对各类训练句在各自原始向量空间做 KMeans，再从每簇选平均簇内余弦相似度最高的真实句子（medoid）。默认每类 15 个。测试/验证句和手动探针不参与。
- 易混淆：同类最佳训练句分数 − 异类最佳训练句分数 &lt; 0.08。训练句排除自身；负值提示异类更近，不等于必然误判。

**不能根据二维簇中心直接选锚点，也不能把二维距离用于新句分类。** 否定句、多意图句、依赖上下文的控制句可能仍混淆；图里分离得漂亮也不保证线上路由可靠。

## 测试报告怎么读

三套方案均按当前“每类取最高余弦相似度，低分或小差值回退聊天”的算法比较：原有 88 条、扩充训练句 500 条、候选锚点默认 75 条。

每套分别显示：

- 现有阈值 `0.62 / 0.03` 的独立测试结果。
- 仅在 100 条验证句上选出的候选阈值，以及该候选在 150 条测试句上的结果。候选在测试集上不一定更好，页面不会隐藏这种退步。
- 测试准确率、Macro F1、每类指标、混淆矩阵、拒绝数量、聊天误触发非聊天意图数量和误判原句。

另有当前会话的“人工标注探针”报告。它与独立测试集分开；未标注探针没有准确率，不能把模型自己的预测当成真值。近重复/合成数据局限始终在报告中显示。以上任何候选都不会自动写回生产配置。

## 分析输出与下载

网页可下载案例、候选锚点和完整 JSON 报告。JSON 是机器可复用的导出格式；页面本身是中文可读的表格和图，不需要阅读 JSON。

`analysis_output/` 中保存：

- `embeddings.npz`：带模型/数据/编码器指纹的向量缓存，数据或模型变更时重新生成。
- `cases.generated.yaml`：分析使用的案例副本。
- `anchors.candidate.yaml`：仅供评估的候选路由样例。
- `report.json`：初始化/重建后的离线分析；网页下载版额外包含当前内存中的探针。
- `large/`：large 独立的向量、报告、候选锚点、案例副本。
- `comparison.json`：双模型对照报告；网页下载版额外包含两边的当前探针。

输出、模型、环境和缓存已加入 `.gitignore`，不会误提交大文件或运行记录。`analysis_lab/cases.yaml`、网页源码、脚本和测试可以提交。没有自动 commit 或 push。

## 验证

```bash
.venv-lab/bin/python -m unittest discover -s tests -v
node --check lab_web/app.js
bash -n setup_lab.sh setup_large_lab.sh run_lab.sh
curl --noproxy '*' -s http://127.0.0.1:8892/api/status
```

测试覆盖数据隔离、唤醒词去重、同环境基线数值一致性、候选锚点只来自训练集、新句真实加入 t-SNE、空标注不伪造准确率、重新分析后探针重新打分、下载、输入校验和跨站保护。原有 13 个路由逻辑测试也保留。

补充双模型真实推理测试覆盖 512/1024 维、独立缓存、设定词/案例完全一致、同句同标签同步探针、两图真实重算、失败原子回滚和按模型下载。`node tests/test_lab_ui.cjs` 用 mock DOM / Plotly 验证双图、切换模型、转义和消息流程，不等于实际浏览器视觉验收。

图表通过本地 Plotly bundle 提供，无 CDN。浏览器支持 WebMCP 时，会注册一个可选的 `analyze_intent_utterances` 工具，调用与表单相同的分析流程；不支持时正常用表单。不依赖该能力，尚未在原生 WebMCP 浏览器上下文中验证。

方法参考：[scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)、[KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)、[t-SNE 读图局限](https://distill.pub/2016/misread-tsne/)。

## 冻结 BGE 的轻量分类器

页面保留原有三个向量方案，新增 `classifier` 对照方案。两个编码器仍使用相同设定词、750 条合成句、数据划分与聚类方式；不微调 BGE，也不增加 GPU 或模型下载。

- 用原始 L2 归一化 BGE 向量训练多类 `LogisticRegression`，仅 500 条 train 参与系数拟合。
- `C` 在 `[0.01, 0.1, 1, 10, 100]` 中按 validation Macro F1、聊天误触发、较小 C 顺序选择。
- 用验证集选择 softmax 温度（最小 Log Loss），再分别为 chat/skill/singing/dancing/focus 选择概率和第二类差值门槛。验证接受精度目标为 95%，至少接受 3 条；在此约束下尽量保留正确决策。不满足时该意图禁用直接接受。
- 报告同时展示裸分类（default）和温度校准 + 拒识（calibrated）。手动探针展示拒识版及裸预测；概率不是余弦，数值不能混用。
- 不确定输出 `clarify`，仅提示澄清，不执行动作；不回退为 chat。真实的机器人澄清流程尚未接入。
- 全量准确率把需澄清算未判对；另报拒识率、动作召回率、接受部分准确率和聊天误触发数量。向量方案仍沿用低分回退聊天，不能把这两种拒识语义混为一谈。

C、温度与门槛复用 100 条小验证集，验证成绩有选择偏差，95% 只是约束目标而不是生产保证。原合成测试集已多次查看，仅用于回归与方法比较；不可当新盲测。BGE 未改变，因此轻量分类器不会改变原有 t-SNE 点位或锚点。

`analysis_output/classifier.candidate.json` 与 `analysis_output/large/classifier.candidate.json` 保存可复用 JSON 系数、截距、类别顺序、温度、各类门槛、训练/验证 ID 与数据哈希、模型指纹。没有 pickle 或反序列化可执行对象；启动时按当前数据重训轻量头以防使用过期参数。候选结果保存在输出目录，不会覆盖 `routes.yaml`、`app.py`、`standalone_app.py` 或更新运行中的路由服务，没有自动 commit/push。

## 真实 ASR 数据与边界标注

“真实 ASR 数据”页可选 JSON 文件或粘贴完整 JSON，支持 1–1000 条、每句不超过 256 字。下载模板为空，不把生成的句子冒充真实记录。JSON 顶层格式为：

```json
{"dataset_id":"采集批次名称","cases":[]}
```

向 cases 填入人工确认的记录，各字段含义如下：

| 字段 | 含义 |
| --- | --- |
| id | 批次内唯一 ID |
| text | 原始 ASR 转写（建议脱敏） |
| mode | chat / skill / singing / dancing / focus；由人标注，不能直接采用模型预测 |
| split | train / validation / test |
| family | 同会话或相近表达家族的稳定唯一 ID；同家族仅放一个集合，跨批次也不可把测试家族移入训练 |
| source | 固定 asr；来源由上传者保证 |

重点收集同主题的对比：讨论唱歌 vs 请求唱歌，询问家务方法 vs 要求执行家务，拒绝执行 vs 停止/继续控制。不能一见“不要/不用”就标为聊天；例如“停止跳舞”是控制舞蹈的请求。需要上下文的原句应人工确认，无法确定的先留在未标注探针，不强贴标签。

只上传 test 即可评测任意小批次，缺类时不作为完整验收。若上传 train 或 validation，该集合五类各至少 5 条；若上传 train，必须另有独立真实 validation。推荐训练每类 100–200 条、验证和留出各类至少 50 条，并按会话/表达家族隔离。

真实 train 仅补充分类器训练集；真实 validation 若存在则替代合成验证集调 C、温度与门槛；真实 test 从不参与拟合。原有向量三方案、合成语料、锚点与散点图保持不变，真实报告单列，明确分类器训练数据数量。测试与训练精确重复、跨集合家族、重复 ID/标准化原句会拒绝，语义训练近重复会告警。

上传是完整当前真实数据集，会替换当前批次，不自动合并；保留需要继续训练的原句时请一并上传。上一份数据自动归档到 `analysis_lab/private/history/`。新批次测试留出应该重新采集；曾评测的测试句与家族 ID 登记为 SHA256，永久禁止再移入训练/验证。同一批次或已查看家族重复查看标为回归，不称作新盲测。

ASR 数据与测试登记保存到 `analysis_lab/private/real_dataset.json`、`test_registry.json`（文件权限 600，目录 700，Git 忽略），重启保留；两模型准备成功后才同时提交，失败不单边生效。真实报告也可能含原句，只保存在被忽略的本地 `analysis_output/`，请不要随意分享含个人信息的下载报告。没有提供真实 ASR 数据时，页面不显示生产准确率。

新增接口：`POST /api/real-dataset` 接收 `{"dataset":"完整JSON字符串"}`；下载 `classifier.candidate.json?model=small|large`、`real-dataset.template.json`；state/comparison/report 包含分类头摘要及分开的真实报告。接口仍只绑定本机。

实现依据：[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)、[概率校准与独立数据要求](https://scikit-learn.org/stable/modules/calibration.html)、[阈值调优的数据隔离](https://scikit-learn.org/stable/modules/classification_threshold.html)。

## 网页人工纠错闭环

不用手写 JSON：

1. 在左侧发送原句，查看两模型分类。选择已发送句或点击历史/散点，打开“判错了？确认标签并归档”。
2. 人工选择正确意图、train/validation/test 和来源。正确案例也能归档。未标注时不会采用模型自己的预测；来源默认 manual，不把网页打字冒充真实语音。
3. 相近改写或同会话填相同稳定 family ID。留空则按标准化原句 SHA256 自动生成单句家族；系统无法自动识别所有表达家族，所以此项仍需人确认。
4. 点击“确认标签并归档”：只保存纠错草稿，并更新两模型中该探针的人工期望标签；不更新系数。
5. 打开“纠错训练”，点“用归档数据重新训练 small / large”。两个候选头全部准备成功后才同时提交，失败恢复原模型与活动数据。原来的向量三方案、750 条散点案例、锚点、生产路由均不改。
6. 看训练前后全量准确率、聊天误触发、动作召回和拒识率。退步照样显示。没有人工标签/留出数据时不伪造准确率；不能把同一批反复查看当新盲测。

草稿 `analysis_lab/private/feedback.json` 与活动快照 `feedback.active.json` 分开。草稿重启仍保留，但不自动训练；重启仅重训上次已激活的记录。训练前后历史对照保存在 `feedback.training.json`。修改归档标签后需再次点训练，页面会提示“有未应用的纠错”。归档集合不可切换；回归留出标签、来源和家族冻结，防止按模型结果改测试真值。训练/验证记录可编辑人工标签与来源；旧版本在 private/history 中可恢复，没有删除接口。

纠错 train 补充原 500 条训练；纠错 validation 补充原验证集（若有导入的真实 validation，补充该集合）。少量纠错也能训练，不要求在网页先录齐五类；但验证中已有五类是由原数据保证，不代表新域完整验收。推荐分会话采集每类至少 100 条，训练/验证/真实留出分别覆盖五类及边界正负例。

纠错 test 只做回归评测，因为这些句子通常已经显示过预测，明确不称作新盲测。真实 ASR 数据页的独立留出另行评测。固定合成验证/测试句、过去测试句或测试家族、跨集合家族、与导入语料重复的原句会拒绝进入纠错训练。ASR 导入也检查与纠错草稿/活动集的精确重复与家族冲突；纠错回归留出与训练余弦 ≥ 0.97 时告警（不能保证没有语义泄漏）。

数据、历史、训练记录和导出报告均在 Git 忽略的本地目录；private 目录权限 700、文件 600，analysis_output 同样限制目录/报告权限。不要分享含个人信息的报告；归档最多 1000 句，不自动上传、commit、push 或部署机器人。

接口：`GET /api/feedback` 获取草稿/活动状态/历史训练对照；`POST /api/feedback` 接收 text/mode/split/source（manual 或 asr）/可选 family；`POST /api/feedback/train` 激活全部草稿并重训两模型；下载 feedback.json 和 feedback.training.json。原 ASR JSON 导入方式仍可使用，且保留已激活的纠错数据。
