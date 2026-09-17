# AENIN

## Adaptive Edge-Node Interaction Network

AENIN is an ongoing research project investigating adaptive learning from
graph-structured brain connectivity data.

The project studies how information associated with graph nodes and their
relationships can be jointly considered for learning representations from
brain-network data.

> **Research status:** Active research. The complete methodology, model
> implementation, experimental configuration, and reproducibility pipeline
> will be released after acceptance of the associated research article.

## Repository status

This repository is currently a **research preview**. It intentionally does not
contain the unpublished AENIN interaction mechanism or the complete
experimental pipeline.

The code currently provided contains only non-sensitive utilities and a public
interface showing the expected form of graph inputs. It should **not** be used
as the implementation required to reproduce the research results.

## Research scope

AENIN is being investigated for graph-based neuroimaging analysis using
resting-state functional brain data. Current work includes representation
learning, graph-level prediction, evaluation, and interpretation of learned
brain-network information.

Detailed implementation choices and experimental settings are intentionally
withheld while the associated research is under review.

## Included in this public preview

- A minimal Python package
- A graph-input validation interface
- Reproducibility utilities
- General classification metric utilities
- A small synthetic-data example
- Tests for the public utilities

## Not included at this stage

The following research components are intentionally not released before
acceptance:

- the complete AENIN architecture;
- the adaptive edge-node interaction mechanism;
- research-specific feature construction;
- graph-generation methodology;
- exact preprocessing and experimental settings;
- trained model parameters;
- complete training and validation code; and
- unpublished analysis and interpretation procedures.

## Data

Original neuroimaging data are not distributed with this repository. Data used
in the research must be obtained from their respective official sources and
used according to the applicable access and data-use requirements.

## Installation

Clone the repository and install the small set of dependencies used by the
public preview:

```bash
git clone https://github.com/YOUR_USERNAME/AENIN.git
cd AENIN
pip install -r requirements.txt
```

## Public preview example

```bash
python examples/synthetic_graph_demo.py
```

This example only demonstrates the expected graph-data interface. It does not
implement the unpublished AENIN method.

## Code availability

The implementation of AENIN is part of ongoing research and is not publicly
available at this stage. The source code and supporting documentation necessary
to reproduce the reported experiments will be made publicly available after
acceptance of the associated research article.

Until then, this repository provides only high-level project information and
non-sensitive supporting code.

## Citation

Citation information will be added after publication of the associated
research article.

## Disclaimer

The research design and implementation may evolve while the work is under
review. Content in this repository should therefore be treated as a research
preview rather than a finalized technical specification.
