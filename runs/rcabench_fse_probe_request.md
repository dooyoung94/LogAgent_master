# RCABench FSE independent-confirmation probe

This marker triggers a metadata-first probe of the official frozen dataset:

- repository: `HamsterStation/rcabench-fse`
- revision: `v1.0.0`
- split: official `manifests/test.txt`
- expected test incidents: 296
- expected `manifests/all.txt` SHA-256: `588165081264b3de851b7b3b7ec59d27a5258107ad04c56d8c1c3f2a5ff76da7`

The probe chooses one test entry from its opaque identifier before reading telemetry, labels, fault type, or model performance. It downloads only trace and metadata files required to freeze the adapter schema.
