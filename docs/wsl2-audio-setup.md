# WSL2 麦克风录音配置

> **本机已完成：** 2026-03-15 本机已按本文档完成全部设置，无需重复执行。

WSL2 + WSLg 下 sounddevice 需 PortAudio + PulseAudio。按以下步骤配置：

**1. Windows：** 设置 → 隐私与安全 → 麦克风 → 开启「允许桌面应用访问麦克风」

**2. WSL2 安装包：**

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 libasound2-plugins alsa-utils pulseaudio-utils
```

**3. 创建 `~/.asoundrc`：**

```bash
cat > ~/.asoundrc << 'EOF'
pcm.!default { type pulse fallback "sysdefault" hint.description "Default Audio Device (via PulseAudio)" }
ctl.!default { type pulse fallback "sysdefault" }
EOF
```

**4. 设置 Pulse 服务器（加入 `~/.bashrc` 并 `source`）：**

```bash
export PULSE_SERVER=unix:/mnt/wslg/PulseServer
```

**5. 桌面模式额外依赖：** `sudo apt install libsecret-1-0`

**验证：** `python3 -c "import sounddevice as sd; print(sd.query_devices())"` 应输出设备列表。
