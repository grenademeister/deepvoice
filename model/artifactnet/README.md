# ArtifactNet v9.4

- Source: https://huggingface.co/intrect/artifactnet
- Pinned revision: `7c9b753a9d006b48e4bfaf85bf0157e135f4aad4`
- Retrieved: 2026-08-28
- Files: `artifactnet_v94_full.onnx`, `artifactnet_v94_full.onnx.data`
- Input documented upstream: mono float32, 44.1 kHz, 4 seconds (`[1, 176400]`)
- Song aggregation documented upstream: median over chunks

Verify the local artifacts with:

```bash
cd model/artifactnet
sha256sum -c SHA256SUMS.txt
```

## License and patent notice

The upstream model card declares the ONNX build licensed under CC BY-NC 4.0 for non-commercial research, academic, and personal evaluation. It also states that patent applications are pending in Korea and through the PCT, and that the CC license does not convey patent rights for commercial deployment.

See `LICENSE.txt` and the upstream model card before use or submission.
