# RoboHarness 离线主页

直接双击 **index.html** 或 **open.cmd**，在 Edge / Chrome 中浏览。
无需联网，无需 Python，无需安装 RoboHarness。网页的图片、PDF 原图、
任务说明、成绩、Method 演示视频和已有的六段 head-camera 视频都在此文件夹中。
没有录制的三个任务保留原网页的提示。GitHub、官方任务网站等外部链接
需要联网，浏览本页及其视频不需要。

## 同步

12023 是更新来源，laptop 保存完整的本地副本。服务器上的目录是
`/home/bince/roboharness_homepage`，laptop 的目标目录是
`C:\Users\16593\Desktop\robotics\roboharness_homepage`。

- 双击 **sync.cmd**：立即同步一次，只下载改变的文件。
- 双击 **enable-sync.cmd**：开启后台自动同步，并在 Windows 登录时启动。
  每分钟检查一次；断网或 12023 不可用时保留离线页面，下次连接时重试。
- 双击 **disable-sync.cmd**：关闭自动同步，保留已下载的页面。

同步使用 Windows 自带的 OpenSSH，默认连接与你提供的登录命令一致：

```powershell
ssh -p 12023 bince@221.12.22.151
```

复用 laptop 现有的 SSH 密钥或 SSH agent。自动同步需要非交互式密钥登录；
首次下载时，本地 Codex 应检查既有的 SSH 配置并验证主机。
如果使用其他 SSH 配置名称，在 PowerShell 中指定：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\sync.ps1 -Install -Server "你的12023连接名称" -Port 0
```

自动同步日志在 `.sync.log`。不需要管理员权限。自动更新的是这个文件夹
中由 12023 发布的页面文件；个人新增文件不会被删除。同步脚本不包含
密码或私钥。移动整个文件夹后，请重新运行 **enable-sync.cmd**。

服务器的 `roboharness-homepage-publisher.service` 每 30 秒检查主页源码
和 Method 视频目录，有变化就更新这个离线文件夹和 ZIP。检查状态：

```bash
systemctl --user status roboharness-homepage-publisher.service
```

需要手动重建时，在 12023 上运行：

```bash
cd /mnt/nas_nfs/home/bince/RoboHarness
python3 scripts/package_site.py --output /home/bince/roboharness_homepage --zip /home/bince/roboharness_homepage.zip
```

## Method 视频

页面顺序为 Abstract → Method → Task results。Method 只有标题和一个视频。
当前用 `roboharness_promo_v5.mp4`；后续自动选取以下目录中版本编号最大的
成品 `roboharness_promo_vN.mp4`：

```text
/mnt/nas_nfs/home/bince/BEHAVIOR-1K/video/demo_edits/roboharness_promo_20261005
```

编辑工程、预览图和仍在写入或解码失败的视频不会替换已有版本。
导出完成后，12023 会更新离线包，并通过独立 Git checkout 发布到 GitHub Pages。
laptop 的同步脚本会拉取新的 Method 视频；可在
`assets/method/manifest.json` 查看当前源文件名和 SHA-256。
浏览器使用本地 `assets/method/roboharness-method.mp4`，断网也能播放。

## laptop 首次下载

压缩包在 12023 的 `/home/bince/roboharness_homepage.zip`，解压后自带
`roboharness_homepage/` 这一层目录。在 laptop PowerShell 中：

```powershell
scp -P 12023 bince@221.12.22.151:/home/bince/roboharness_homepage.zip "$env:TEMP\roboharness_homepage.zip"
Expand-Archive -LiteralPath "$env:TEMP\roboharness_homepage.zip" -DestinationPath "C:\Users\16593\Desktop\robotics"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\16593\Desktop\robotics\roboharness_homepage\sync.ps1" -Install
```

已有同名文件夹时，请先检查并备份需要覆盖的文件再更新。
完整的本地 Codex 任务见 **LOCAL_CODEX_PROMPT.md**。

## 完整性

`offline-manifest.json` 列出每个文件的大小和 SHA-256。同步脚本先下载并
校验新文件，再替换本地文件；下载失败不会替换对应的已有文件。
`licenses/` 保存 MIT 许可证和第三方说明，可离线查看。
