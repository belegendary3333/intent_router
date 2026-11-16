# Coder 后台服务部署

本模板适用于已提供 `coder-svc` 命令和 Supervisor 用户服务管理能力的 Coder 工作区。请先确认工作区支持 `~/.config/coder/services/` 配置目录；不同工作区镜像可能需要调整服务管理方式。

## 工作区与持久化路径

示例将项目放在 `$HOME/intent_router`，配置通过 `%(ENV_HOME)s` 引用实际用户主目录。部署前确认项目、模型和运行环境所在目录使用工作区的持久化存储；如果项目位于其他目录，请对应修改配置中的 `command` 和 `directory`。

- 源码、`.env`、`model/` 和 `.offline-site/`：`~/intent_router/`。
- 配置模板：`~/intent_router/deploy/coder/intent_router.conf`。
- Supervisor 实际加载：`~/.config/coder/services/intent_router.conf`。
- 托管日志：`~/.local/state/coder-services/intent_router.log`，单文件 10 MB、保留 3 份轮转备份，stderr 合并到同一个文件。

启动脚本 `run_container.sh` 必须保持前台 `exec` 运行。不要在里面添加 `nohup`、`&` 或 daemon 参数。配置使用 `%(ENV_HOME)s`，由 Supervisor 展开真实 HOME。

## 首次接入

先确认项目已放在 `~/intent_router`，离线依赖和 ONNX 模型齐全，`.env` 保留 `ROUTER_HOST=127.0.0.1` 和 `ROUTER_PORT=8890`。

```bash
mkdir -p "$HOME/.config/coder/services" "$HOME/.local/state/coder-services"
cp -i "$HOME/intent_router/deploy/coder/intent_router.conf" \
  "$HOME/.config/coder/services/intent_router.conf"
coder-svc reread
```

如果端口 8890 上还有旧的手工后台进程，先核对其 PID、完整命令行、工作目录和所属用户，再对该进程发送 TERM 并等待退出。不要按模糊名称 `pkill`，不要仅依赖旧的 PID 文件；已有服务必须先停，避免两个进程争抢端口。

然后只应用本服务，不更新其他组：

```bash
coder-svc update intent_router
coder-svc status intent_router
curl --noproxy '*' -fsS http://127.0.0.1:8890/health
/usr/bin/python3 "$HOME/intent_router/smoke_test.py"
```

## 日常使用

```bash
coder-svc status intent_router
coder-svc restart intent_router
coder-svc stop intent_router
coder-svc start intent_router
tail -n 100 -f "$HOME/.local/state/coder-services/intent_router.log"
```

修改 `.env`、代码或 `routes.yaml` 后使用 `restart`。修改 Supervisor 配置后使用 `coder-svc reread` 和 `coder-svc update intent_router`。

`autostart=true` 使服务在工作区启动时启动；`autorestart=true` 使意外退出的服务自动重启。主动执行 `coder-svc stop intent_router` 会保持停止，不会立即被重新拉起。此配置不会阻止 Coder 平台停止整个工作区，也不会让已停止的工作区继续提供 API。

Supervisor 托管后的输出在上述日志中，进程由服务管理器维护。如果此前手动运行过服务，可按需保留旧日志用于排查。

## 网络访问范围

示例服务仅监听 `127.0.0.1:8890`，供同一工作区内的客户端调用。跨机器调用需要配置可达的网络入口及访问控制，并在客户端设置对应服务地址。路由 API 本身没有认证，部署时请限制访问范围。

## 迁移回退

需要退出 Supervisor 托管时，先 `coder-svc stop intent_router`，把实际加载的 `.conf` 移至 `~/intent_router/deploy/coder/intent_router.conf.disabled` 留档，再 `coder-svc reread` 和 `coder-svc update intent_router`。确认端口释放后，可在前台执行 `bash "$HOME/intent_router/run_container.sh"`。不要同时运行两套启动方式。
