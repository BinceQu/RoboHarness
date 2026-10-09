# Uses the laptop's existing OpenSSH connection; no password or key is bundled.
[CmdletBinding()]
param(
    [string]$Server = "bince@221.12.22.151",
    [int]$Port = 12023,
    [string]$RemoteDirectory = "/home/bince/roboharness_homepage",
    [int]$Interval = 60,
    [switch]$Watch,
    [switch]$Install,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$folder = $PSScriptRoot
$settingsPath = Join-Path $folder ".sync-settings.json"
$stopPath = Join-Path $folder ".sync-stop"
$logPath = Join-Path $folder ".sync.log"
$shortcutPath = Join-Path ([Environment]::GetFolderPath("Startup")) "RoboHarness Homepage Sync.lnk"
if (Test-Path $settingsPath) {
    $saved = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
    if (!$PSBoundParameters.ContainsKey("Server")) { $Server = $saved.Server }
    if (!$PSBoundParameters.ContainsKey("Port")) { $Port = $saved.Port }
    if (!$PSBoundParameters.ContainsKey("RemoteDirectory")) { $RemoteDirectory = $saved.RemoteDirectory }
}
if ($Server -notmatch '^[A-Za-z0-9][A-Za-z0-9_.@:-]*$' -or $Port -lt 0 -or $Port -gt 65535) {
    throw "Invalid SSH server or port."
}
if ($RemoteDirectory -notmatch '^/[A-Za-z0-9._/-]+$' -or ($RemoteDirectory -split '/') -contains '..') {
    throw "Use an absolute remote folder without spaces or parent traversal."
}
$RemoteDirectory = $RemoteDirectory.TrimEnd('/')
if ($Interval -lt 5) { throw "Use a sync interval of at least five seconds." }

if ($Uninstall) {
    Remove-Item -LiteralPath $shortcutPath -Force -ErrorAction SilentlyContinue
    Set-Content -LiteralPath $stopPath -Value "stop" -Encoding ASCII
    Write-Host "Automatic sync disabled. The offline page is retained."
    exit 0
}

$options = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ConnectionAttempts=1',
    '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', '-o', 'StrictHostKeyChecking=yes', '-o', 'LogLevel=ERROR')
$sshOptions = @($options)
$scpOptions = @($options)
if ($Port) { $sshOptions += @('-p', "$Port"); $scpOptions += @('-P', "$Port") }
$sshProgram = (Get-Command ssh.exe -ErrorAction Stop).Source
$scpProgram = (Get-Command scp.exe -ErrorAction Stop).Source
$hash = [Security.Cryptography.SHA256]::Create()
$folderHash = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($folder)))).Replace('-', '')
$hash.Dispose()

function Write-SyncLog([string]$message) {
    if ((Test-Path $logPath) -and (Get-Item -LiteralPath $logPath -Force).Length -gt 1048576) {
        $tail = Get-Content -LiteralPath $logPath -Tail 200
        Set-Content -LiteralPath $logPath -Value $tail -Encoding UTF8
    }
    Add-Content -LiteralPath $logPath -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $message) -Encoding UTF8
}

function Sync-Once {
    $mutex = New-Object Threading.Mutex($false, ("Local\RoboHarnessHomepageCopy_" + $folderHash))
    $locked = $false
    try {
        $locked = $mutex.WaitOne(0)
        if (!$locked) { return }
        $raw = & $sshProgram @sshOptions $Server "cat '$RemoteDirectory/offline-manifest.json'"
        if ($LASTEXITCODE -ne 0) { throw "Cannot reach 12023 through SSH. Check the existing SSH alias or specify -Server and -Port." }
        $json = $raw -join "`n"
        $manifest = $json | ConvertFrom-Json
        if ($manifest.version -ne 1 -or !$manifest.files -or $manifest.contentHash -notmatch '^[a-f0-9]{64}$') {
            throw "Invalid offline manifest."
        }
        $changed = 0
        $staging = Join-Path $folder ".sync-staging"
        New-Item -ItemType Directory -Path $staging -Force | Out-Null
        foreach ($entry in $manifest.files.PSObject.Properties) {
            $relative = $entry.Name
            if ($relative -notmatch '^[A-Za-z0-9._/-]+$' -or $relative.StartsWith('/') -or
                ($relative -split '/') -contains '..' -or $entry.Value.sha256 -notmatch '^[a-f0-9]{64}$') {
                throw "Invalid file in offline manifest."
            }
            $target = Join-Path $folder ($relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
            if ((Test-Path -LiteralPath $target -PathType Leaf) -and
                (Get-Item -LiteralPath $target -Force).Length -eq $entry.Value.bytes -and
                (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -eq $entry.Value.sha256) {
                continue
            }
            $incoming = Join-Path $staging "incoming"
            & $scpProgram -q @scpOptions ("{0}:{1}/{2}" -f $Server, $RemoteDirectory, $relative) $incoming
            if ($LASTEXITCODE -ne 0) { throw "Download failed: $relative. The offline copy has been retained." }
            if ((Get-Item -LiteralPath $incoming -Force).Length -ne $entry.Value.bytes -or
                (Get-FileHash -LiteralPath $incoming -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value.sha256) {
                throw "The downloaded file changed during sync; the next sync will retry: $relative"
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Move-Item -LiteralPath $incoming -Destination $target -Force
            $changed++
        }
        $temporaryManifest = Join-Path $staging "offline-manifest.json"
        [IO.File]::WriteAllText($temporaryManifest, $json + "`n", (New-Object Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temporaryManifest -Destination (Join-Path $folder "offline-manifest.json") -Force
        $message = "Up to date. Updated $changed file(s)."
        if (!$Watch) { Write-Host $message }
        if ($changed) { Write-SyncLog $message }
    } finally {
        if ($locked) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}

if ($Install) {
    @{ Server = $Server; Port = $Port; RemoteDirectory = $RemoteDirectory } |
        ConvertTo-Json | Set-Content -LiteralPath $settingsPath -Encoding UTF8
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $PSHOME "powershell.exe"
    $shortcut.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + (Join-Path $folder 'sync.ps1') + '" -Watch'
    $shortcut.WorkingDirectory = $folder
    $shortcut.WindowStyle = 7
    $shortcut.Save()
    Remove-Item -LiteralPath $stopPath -Force -ErrorAction SilentlyContinue
    Start-Process -FilePath $shortcut.TargetPath -ArgumentList $shortcut.Arguments -WindowStyle Hidden
    Write-Host "Automatic sync enabled now and at Windows sign-in. Checks 12023 every 60 seconds."
    Write-Host "Double-click index.html or open.cmd to browse offline."
    exit 0
}

if ($Watch) {
    $watchMutex = New-Object Threading.Mutex($false, ("Local\RoboHarnessHomepageWatch_" + $folderHash))
    $watchLocked = $false
    try {
        $watchLocked = $watchMutex.WaitOne(0)
        if (!$watchLocked) { exit 0 }
        while (!(Test-Path -LiteralPath $stopPath)) {
            try { Sync-Once } catch { Write-SyncLog $_.Exception.Message }
            for ($elapsed = 0; $elapsed -lt $Interval -and !(Test-Path -LiteralPath $stopPath); $elapsed++) {
                Start-Sleep -Seconds 1
            }
        }
    } finally {
        if ($watchLocked) { $watchMutex.ReleaseMutex() }
        $watchMutex.Dispose()
    }
} else {
    Sync-Once
}
