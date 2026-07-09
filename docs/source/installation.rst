Installation
============

Footix requires PyTorch. By default, the **CPU-only** version is installed to avoid
downloading large CUDA packages (~2 GB on Linux).

CPU (default)
-------------

**uv** (recommended) — CPU torch is resolved automatically via the configured index:

.. code-block:: bash

   uv add pyfootix

**pip** — pass the PyTorch CPU wheel index to avoid CUDA packages:

.. code-block:: bash

   pip install pyfootix --extra-index-url https://download.pytorch.org/whl/cpu

GPU (Linux with CUDA)
---------------------

If you need GPU acceleration (e.g. for portfolio optimization with large datasets),
install from standard PyPI which includes CUDA support on Linux:

.. code-block:: bash

   # pip — standard PyPI includes CUDA on Linux
   pip install pyfootix

   # uv — override the configured CPU index to use PyPI torch
   uv add pyfootix --no-sources

The ``gpu`` extra installs ``triton`` (GPU kernel compiler):

.. code-block:: bash

   pip install "pyfootix[gpu]"
