# 交给 laptop 本地 Codex 的任务

请直接执行，把 RoboHarness 主页从 12023 下载到 Windows laptop，并设置持续同步和离线浏览。

已授权你创建目标文件夹、下载网页文件、配置当前用户的后台同步和登录启动项。

- 目标目录：`C:\Users\16593\Desktop\robotics\roboharness_homepage`
- 12023 连接：`ssh -p 12023 bince@221.12.22.151`
- 服务器文件夹：`/home/bince/roboharness_homepage`
- 完整压缩包：`/home/bince/roboharness_homepage.zip`，含 Method 演示和六段任务视频，内部含 `roboharness_homepage/` 顶层目录。

先检查本机已有的 SSH 配置和 SSH agent，复用现有登录方式，不输出私钥或密码。
检查目标目录；若有同名文件，先备份会被覆盖的文件，保留个人新增文件。
使用 `scp -P 12023` 下载压缩包，在 `C:\Users\16593\Desktop\robotics` 解压，
避免变成 `roboharness_homepage\roboharness_homepage` 两层目录。

阅读包内 `README-offline.md`，用 `offline-manifest.json` 校验所有文件的大小和 SHA-256。
随后执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\16593\Desktop\robotics\roboharness_homepage\sync.ps1" -Install
```

同步方向为 12023 → laptop，每分钟检查一次，只传变化的文件；断网时保留本地副本，
恢复连接后继续同步。脚本默认使用 `bince@221.12.22.151` 和端口 `12023`。
若现有 SSH 配置需指定密钥等选项，使用它的 Host 别名，并给脚本传 `-Server 别名 -Port 0`。
验证非交互式 SSH 登录是否可用；如果只有密码登录，说明这一具体阻塞，
不要把密码写入文件，也不要声称自动同步已成功。

测试一次同步，并确认没有重传未变化的视频。查看 `.sync.log` 和当前用户的
`RoboHarness Homepage Sync.lnk` 登录启动项，确认后台同步已启动。
双击 `index.html` 或 `open.cmd` 打开网页。临时阻止浏览器访问外网，
验证 Abstract 后、Task results 前的 Method 演示视频（当前 v5）、100 objects 图片、
九张任务卡片的说明和成绩、六段任务视频播放与
左右箭头切换都可用；另外三张卡片原本就没有录像。浏览本页不应依赖 CDN、
GitHub、Python 或本地服务器。GitHub 和官方任务的外部链接需联网，这不影响离线浏览。
确认 `assets/method/manifest.json` 所选版本，并检查 Method 视频也被纳入同步清单。
12023 会自动选取指定演示目录里最新的成品版本，更新离线包并发布线上页面。

完成后报告实际下载目录、完整性校验和离线浏览结果、自动同步状态，
以及如何用 `disable-sync.cmd` 关闭同步。只有实际验证过的步骤才能报告为成功。
