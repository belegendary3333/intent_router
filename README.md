# BGE Intent Router

基于 BGE 中文文本向量和 ONNX 的轻量意图路由服务，提供 FastAPI 接口及本地可视化分析台。输入一句中文，返回意图类别、匹配样例、相似度及是否通过阈值判断；下游系统可据此决定如何处理请求。

| 意图 | 含义 | 路由样例数 |
| --- | --- | ---: |
| `chat` | 聊天、知识讨论 | 24 |
| `skill` | 任务与动作请求 | 19 |
| `singing` | 唱歌 | 21 |
| `dancing` | 跳舞 | 16 |
| `focus` | 专注模式 | 8 |

当前 [routes.yaml](routes.yaml) 共包含 88 条样例。服务仅返回意图，不执行机器人动作。

## 功能与工作方式

- **路由 API**：使用 `BAAI/bge-small-zh-v1.5` 的 512 维归一化向量，对每个类别取最佳样例相似度。最佳分数至少为 `0.62`，且与第二类别的分差至少为 `0.03` 时接受，否则回退为 `chat`。
- **本地分析台**：输入原句、查看交互式 t-SNE 图和代表锚点，比较原有样例、扩充样例、锚点及冻结 BGE 的轻量分类器。
- **模型对照与纠错**：可选下载 BGE large，在同一数据集上比较 small / large；支持导入标注 ASR 数据和网页人工纠错，再训练实验分类头。

分析台使用五类各 150 条、共 750 条合成案例，按表达家族隔离训练、验证和测试集。合成数据结果用于方法比较，不能替代真实业务数据验收；分析台的训练和候选导出不会自动改写 `routes.yaml`。

## 快速开始：路由服务

推荐 Python 3.11；安装脚本支持 Python 3.10–3.12。首次安装需要联网下载 Python 依赖与模型，模型文件不包含在 Git 仓库中。

```bash
git clone https://github.com/belegendary3333/intent_router.git
cd intent_router

bash setup.sh
bash run.sh
```

`setup.sh` 创建独立 `.venv`、安装 [requirements.txt](requirements.txt)，在 `.env` 不存在时复制 [.env.example](.env.example)，并下载、导出默认 FP32 ONNX 模型。已安装 `uv` 时使用 `uv`；否则需要 `python3.11`，也可通过 `ROUTER_PYTHON=/path/to/python` 指定兼容解释器。

服务默认地址为 `http://127.0.0.1:8890`，交互接口文档为 `http://127.0.0.1:8890/docs`。前台运行，按 `Ctrl+C` 停止；修改代码、`.env` 或 `routes.yaml` 后需重启。启动脚本使用一个 worker，避免重复加载模型。

### 调用接口

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8890/health

curl --noproxy '*' -fsS http://127.0.0.1:8890/route \
  -H 'Content-Type: application/json' \
  -d '{"text":"小诺，给我唱首歌"}'
```

`GET /health` 返回服务状态、模型名、运行后端、向量维度及样例数量。`POST /route` 接收包含 `text` 的 JSON 对象，文本长度为 1–256 个字符，返回字段如下：

| 字段 | 说明 |
| --- | --- |
| `text` | 去除首尾空白和开头“小诺”唤醒词后的文本 |
| `mode` | 最终意图；未通过阈值时为 `chat` |
| `matched_mode` | 相似度最高的候选类别 |
| `matched_example` | 该类别内匹配度最高的样例 |
| `score` | 最佳相似度；不是正确率或概率 |
| `margin` | 最佳类别与第二类别的分差 |
| `accepted` | 是否同时通过分数和分差阈值 |
| `scores` | 各类别的最佳相似度 |
| `latency_ms` | 本次路由处理耗时，单位毫秒 |

仅包含空白或唤醒词时返回 `chat`、`accepted: false`，并带有 `reason` 字段。缺少 `text`、空字符串、非字符串或超过 256 字符的请求返回 HTTP 422。

### 配置

`run.sh` 读取项目根目录的 `.env`。直接通过 Uvicorn 启动时，需要自行设置相应环境变量。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `ROUTER_HOST` | `127.0.0.1` | 监听地址 |
| `ROUTER_PORT` | `8890` | 监听端口 |
| `ROUTER_MIN_SCORE` | `0.62` | 最低相似度 |
| `ROUTER_MIN_MARGIN` | `0.03` | 最低类别分差 |
| `ROUTER_MODEL_DIR` | 项目内 `model/` | 模型目录 |
| `ROUTER_ONNX_FILE` | `onnx/model.onnx` | 相对于模型目录的 ONNX 路径 |
| `ROUTER_ROUTES_FILE` | 项目内 `routes.yaml` | 路由样例配置 |
| `OMP_NUM_THREADS` | `4` | 线程配置 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

`routes.yaml` 必须包含上述五个类别，每类至少一条非空字符串样例。服务没有内置认证；需要跨机器访问时，应配置受控网络入口。

### 可选：导出 CPU INT8 模型

首次安装默认导出通用 FP32 ONNX。Apple Silicon 可在安装完成后执行：

```bash
.venv/bin/python export_bge.py --quantization arm64
```

然后将 `.env` 中的 `ROUTER_ONNX_FILE` 设置为 `onnx/model_qint8_arm64.onnx` 并重启。x86 AVX2 机器可改用 `--quantization avx2`，对应文件为 `onnx/model_qint8_avx2.onnx`。不同模型文件和运行环境可能产生分数及耗时差异，使用前需在目标环境验证。

## 快速开始：本地分析台

分析台使用独立 `.venv-lab`，默认 Python 3.13；当前锁定依赖要求 Python 3.12–3.13。首次安装需联网下载依赖和已知 SHA256 的公开 INT8 ONNX 文件，之后可在本地离线分析。

```bash
# 在项目根目录执行
bash setup_lab.sh
bash run_lab.sh
```

打开 `http://127.0.0.1:8892/`，等待页面显示分析就绪。没有 `uv` 时应确保 `python3` 为兼容版本，或用 `LAB_PYTHON=/path/to/python` 指定；使用 `uv` 时也可指定 `LAB_PYTHON=3.12`。

分析台和主服务默认共用 `model/`，但使用不同 ONNX 文件。下载器遇到哈希不符的现有文件会拒绝覆盖。如果同时使用两套安装流程，建议为分析台指定独立 small 模型目录，并在安装、运行和测试时使用相同设置：

```bash
export LAB_MODEL_DIR="$PWD/model/lab-small"
bash setup_lab.sh
bash run_lab.sh
```

可选的 large 模型默认保存在 `model/large/`，不替换 small：

```bash
bash setup_large_lab.sh
# 停止后重新运行分析台，自动构建 small / large 对照
bash run_lab.sh
```

`LAB_PORT` 可更改网页端口，例如 `LAB_PORT=8893 bash run_lab.sh`。`LAB_MODEL_DIR` / `LAB_ONNX_FILE` 指定 small 模型，`LAB_LARGE_MODEL_DIR` / `LAB_LARGE_ONNX_FILE` 指定 large 模型；两个 ONNX 文件的默认相对路径均为 `onnx/model_quantized.onnx`。实验脚本读取终端环境变量，不自动加载主服务的 `.env`。若自定义 large 目录，需先用 `download_large_model.py --directory /path/to/large` 下载到相同位置。

报告、向量缓存和候选文件写入 `analysis_output/`，真实标注及纠错数据写入 `analysis_lab/private/`，这些本地数据均不随代码提交。图表资源由本地提供，不依赖 CDN。分析方法、数据划分、报告解释、ASR 导入和纠错流程见 [README_LAB.md](README_LAB.md)。

### 直接使用 ONNX Runtime

[standalone_app.py](standalone_app.py) 提供不依赖 Sentence Transformers 的 API 实现。已完成分析台安装及 small INT8 模型下载后，可使用其环境启动：

```bash
ROUTER_MODEL_DIR="${LAB_MODEL_DIR:-$PWD/model}" \
ROUTER_ONNX_FILE=onnx/model_quantized.onnx \
  .venv-lab/bin/python -m uvicorn standalone_app:app \
  --host 127.0.0.1 --port 8890 --workers 1
```

启动前先停止占用同一端口的服务。[run_container.sh](run_container.sh) 是针对已准备好 `.offline-site/` 的离线容器环境的入口，不负责安装依赖；Supervisor 部署参考见 [deploy/coder/README.md](deploy/coder/README.md)。

## 验证

仅验证主服务的路由逻辑与 HTTP 协议，无需模型权重：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_app.py' -v
```

这些测试使用模拟向量，覆盖样例配置、五类路由、阈值回退、唤醒词处理及请求校验，不能用于评估真实模型精度。

在已启动的真实服务上运行冒烟检查：

```bash
.venv/bin/python smoke_test.py
# 非默认地址可指定：
# .venv/bin/python smoke_test.py --base-url http://127.0.0.1:8891
```

完成分析台安装后，可运行完整 Python 测试与前端语法检查；缺少所需模型时，相应集成测试会跳过：

```bash
.venv-lab/bin/python -m unittest discover -s tests -v
node --check lab_web/app.js
bash -n setup.sh run.sh setup_lab.sh setup_large_lab.sh run_lab.sh run_container.sh
```

前端交互回归测试 `node tests/test_lab_ui.cjs` 还依赖本地 small / large 分析报告及对照报告，需先下载两模型并成功启动分析台完成构建。该测试使用模拟 DOM / Plotly，不等于浏览器视觉验收。

## 项目结构

```text
intent_router/
├── app.py                    # Sentence Transformers ONNX 路由 API
├── standalone_app.py         # 直接使用 ONNX Runtime 的路由 API
├── routes.yaml               # 五类、88 条路由样例
├── export_bge.py              # 模型下载、导出与可选量化
├── setup.sh / run.sh          # 主服务环境安装和启动
├── lab_app.py / lab_encoder.py
├── analysis_lab/             # 合成语料、分类器、评估与纠错逻辑
├── lab_web/                  # 分析台网页资源
├── setup_lab.sh / run_lab.sh
├── setup_large_lab.sh         # 可选 large 模型下载入口
├── requirements*.txt         # 主服务和分析台依赖
├── requirements-lab.lock      # 分析台锁定版本
├── tests/ / smoke_test.py
├── deploy/coder/              # Supervisor 部署参考
├── .env.example              # 主服务配置模板
└── README_LAB.md             # 分析台详细说明
```

[.gitignore](.gitignore) 排除虚拟环境、实际 `.env`、模型权重、日志、分析输出、私有标注和本地备份。克隆仓库后需重新安装依赖、准备模型。
