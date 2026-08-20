# models/

Model weights are downloaded into this folder automatically on first use:

| File | Purpose | Source |
|------|---------|--------|
| `yolo11n.pt` (or the model chosen in the sidebar / `config.yaml`) | YOLO person detector | Ultralytics GitHub release assets |
| `osnet_x1_0_msmt17.pt` (or another OSNet variant) | OSNet Re-ID backbone, 512-d embeddings | torchreid model zoo (Google Drive, via BoxMOT's downloader) |

Supported OSNet weights (auto-download): `osnet_x1_0_{msmt17,market1501,dukemtmcreid}.pt`,
`osnet_ain_x1_0_msmt17.pt`, `osnet_ibn_x1_0_msmt17.pt`, `osnet_x0_75_msmt17.pt`, `osnet_x0_5_msmt17.pt`,
`osnet_x0_25_msmt17.pt`.

If automatic download is impossible (no internet / Google Drive blocked), download the files
manually and place them here with exactly these names. Custom YOLO checkpoints can be referenced
by path in `config.yaml` (`detector.model`).
