# TransFolk Core

`transfolk-core` is the reusable Python core library of the TransFolk ecosystem. It contains the symbolic music processing, tokenization, modelling, training, generation, evaluation and visualization components used by TransFolk-based projects.

This package is designed to be shared by several independent repositories, including the TransFolk backend, TEIMUS and future projects based on symbolic music generation or analysis.

---

## Purpose

The goal of `transfolk-core` is to provide a clean and reusable foundation for working with symbolic music corpora, especially folk melodies encoded as MusicXML, token sequences or model-ready representations.

It includes:

- symbolic music preprocessing;
- Mode–Metric tokenization;
- transformer model definitions;
- training utilities;
- generation utilities;
- musical metrics;
- pattern analysis;
- plotting utilities;
- configuration entities and path resolution tools.

This repository should contain reusable code only. It should not contain full corpora, trained model weights, generated outputs or experiment results.

---

## Author
Brian Martínez-Rodríguez

GitHub: https://github.com/BrianComposer

Email: info@brianmartinez.music

Web: www.brianmartinez.music

## License

MIT License

---

## Repository structure

```text
transfolk-core/
├── src/
│   └── transfolk_core/
│       ├── charts/
│       ├── config/
│       ├── features/
│       ├── generation/
│       ├── metrics/
│       ├── model/
│       ├── patterns/
│       ├── pipeline/
│       ├── preprocessing/
│       ├── tokenization/
│       ├── training/
│       ├── transfolk/
│       └── utils/
├── test/
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md




