# AMR3D-NMRI

Accelerated high-resolution and motion-robust 3D volume reconstruction for neonatal MRI

This repository contains the code used in the thesis project of Maurice Kingma, titled "Accelerated high-resolution and motion-robust 3D volume reconstruction for neonatal MRI". Used to obtain a reconstructed volume with high isotropic resolution of the neonatal brain.

![AMR3D-NMRI-overview](http://github.com/MYKingma/AMR3D-NMRI/blob/main/AMR3D-NMRI-overview.png?raw=true)

_Link to published thesis will be posted when available_

## Requirements

This repoistiry makes use of the [Bart toolbox](https://mrirecon.github.io/bart/) and [MRecon](https://www.gyrotools.com/gt/index.php/products/reconframe) with MRSense from Gyrotools to preprocess the raw MRI data. The [MRIDC](https://github.com/wdika/mridc) toolbox is used for implementing the motion-robust reconstrucion model. The [Slice-to-Volume Reconstruction ToolKit](https://github.com/SVRTK/SVRTK) is used for transforming anisotropic slice stacks isotropic resolution.

## Installation

It is recommended to [use a virtual environment](https://conda.io/projects/conda/en/latest/user-guide/index.html) to install the required packages. The following commands will create a virtual environment and install the required packages:

```bash
conda create -n AMR3D-NMRI python=3.11
conda activate AMR3D-NMRI
pip install -r requirements.txt
```

To enable the import of utilities, add the project root directory to the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/AMR3D-NMRI/code
```

Or, if you are using a Windows machine:

```bash
set PYTHONPATH=%PYTHONPATH%;C:\path\to\AMR3D-NMRI\code
```

When using a supported IDE, such as [Visual Studio Code](https://code.visualstudio.com/), you can also add the python path to the integrated terminal settings:

`.vscode/settings.json`

```json
{
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "C:\\path\\to\\AMR3D-NMRI\\code"
  },
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "/path/to/AMR3D-NMRI/code"
  },
  "terminal.integrated.env.osx": {
    "PYTHONPATH": "/path/to/AMR3D-NMRI/code"
  }
}
```
