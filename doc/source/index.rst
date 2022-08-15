.. DTB documentation master file, created by
   sphinx-quickstart on Sun Aug  7 20:37:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DTB's documentation!
===============================

**DTB** is a super large scale neuromorphic simulation platform.
DTB is a parallel and distributed system, which deploy ``large-scale neuronal networks`` on 10,000 GPU cards,which operates and communicates information in real time.
By optimizing the neuron layout and routing communication of cards,
we realize the 1:1 human brain simulation with a deceleration ratio of 560.
In order to model a biological-plausible brain, we develop a ``hierarchal mesoscale data assimilation (HMDA)`` method to estimate 10 trillion parameters in DTB,
which succeeds to reconstruct a cognitive brain.

.. image:: fig/DTB.png
  :width: 800

.. toctree::
   :maxdepth: 2
   :hidden:

   rstfiles/install
   rstfiles/user_guide
   modules
