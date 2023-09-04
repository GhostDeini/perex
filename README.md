# perex - Python Environment for Relating Electrochemistry and XAS data
## The ECXAS package
In the context of the BIG-MAP (Battery Interface Genome Materials Accelerated Platform, www.big-map.eu) European project, a need for semi-automated data processing in battery characterisation was identified, specially for experiments concerning the use of large-scale facilities. The general efforts in synchrotrons and neutron sources to improve time and spectral resolution, along with the expanded use of multi-modal approaches (involving several techniques in parallel) has been resulting in ever-increasing volumes of data, impossible to be handled manually.

A simple example is the coupling of x-ray absorption spectroscopy (XAS) with electrochemical characterisation (EC). This is possible at ROCK beamline, in Synchrotron SOLEIL. In the current state, the data arrive out of sync at ROCK (for example, time series of absorption spectroscopy and electrochemical potential during a charge-discharge cycle of a battery), which makes their handling and representation complex. This has prompted the need of a tool that aggregates asynchronous data and allows simple visual representations.

ECXAS is a data aggregation tool for battery study in ROCK beamline, at Synchrotron SOLEIL. Here we present an example of data aggregation and visualization for an operando XAS experiment coupled with electrochemical characterization performed at the beamline.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GhostDeini/perex/HEAD?labpath=example%2Fecxas_rock_example.ipynb)

## Installing
Run the command:
pip install git+https://github.com/GhostDeini/perex

## Qcknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 957189. The project is part of BATTERY 2030+, the large-scale European research initiative for inventing the sustainable batteries of the future.
