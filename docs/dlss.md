# NVIDIA DLSS in the launcher

[Français](fr/dlss.md) · [Launcher guide](launcher.md)

**Super Resolution** reconstructs the output image from a lower-resolution render, depth and motion vectors. Quality keeps more scene detail. Balanced, Performance and Ultra Performance progressively reduce the number of pixels rendered. DLAA applies anti-aliasing at the native output resolution.

**Frame Generation 2x** uses motion between rendered frames to insert one intermediate image. The game still samples input at its rendered frame rate. A reference run on **27 November 2026** with the **VIctory-NVIDIA build** on a **GeForce RTX 4070 SUPER** measured **12 frames rendered and 24 displayed per second**. VIctory’s frame counter reports rendered frames. The Streamline log reports frames shown.

## Set up Streamline

1. Use a GeForce RTX card with a recent NVIDIA driver. For Frame Generation 2x, use an RTX 40 or 50 series card. In Windows, open **Settings → System → Display → Graphics**, enable **Hardware-accelerated GPU scheduling** in the default graphics settings and restart the PC.
2. Open **Play → NVIDIA DLSS → Télécharger les fichiers NVIDIA**. The button opens NVIDIA’s releases page in your browser. Download the Streamline SDK archive yourself (reference version **2.14.16**) and extract it outside the repository. NVIDIA supplies the DLLs under its own licences. AnyXboxX ships none of them. [Official NVIDIA Streamline releases](https://github.com/NVIDIA-RTX/Streamline/releases).
3. Under **Streamline folder**, select the extracted release’s `bin\x64` directory. Keep `sl.interposer.dll`, `sl.common.dll`, `sl.dlss.dll`, `sl.dlss_g.dll`, `sl.reflex.dll`, `sl.pcl.dll`, `nvngx_dlss.dll` and `nvngx_dlssg.dll` together. The launcher checks the interposer’s Authenticode signature and requires the signer **NVIDIA Corporation**. If the check fails, confirm the directory and use the release files from NVIDIA.
4. Enable **DLSS Super Resolution** and choose a mode, or enable **Frame Generation 2x**, or use both. A grey Frame Generation switch explains which prerequisite is missing. For 1440p, enable both features because the title’s own FidelityFX upscaler used with frame generation alone supports a limited set of sizes.

## Settings and limits

The launcher saves these choices in `%LOCALAPPDATA%\AnyXboxX\launcher\settings.toml`. Either feature selects the title’s FidelityFX path (`VICTORY_UPSCALER=7`) and uses `VICTORY_STREAMLINE_DIR`. Super Resolution sets `VICTORY_DLSS=1`, `VICTORY_DLSS_MODE` and `VICTORY_DLSS_PRESET`. Frame Generation sets `VICTORY_FRAMEGEN=1`, `VICTORY_THIN=1` and `VICTORY_FRAMEGEN_HUDLESS`. Overrides entered in **Advanced** take precedence. At launch the signature check uses the effective folder.

- **Latency.** Generating an intermediate frame requires holding a rendered frame back. At 12 rendered frames per second, input updates are still about 83 ms apart and buffering can add delay.
- **Window focus.** Generation pauses when the game window loses focus. Bring it to the foreground before comparing the two counters.
- **Interface.** Leave **Exclude the HUD from generated motion** enabled. It uses a copy of the scene taken before the HUD is drawn. Interface elements drawn separately can still distort around moving objects. Set `VICTORY_FRAMEGEN_HUDLESS=0` in Advanced to compare.
