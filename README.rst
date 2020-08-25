========
pyembree
========
Python Wrapper for Embree

Installation
------------

Install embree 3.x, install this. Badabing-badabom.

Suppressing errors
------------------

Creating multiple scenes produces some harmless error messages:
::
    ERROR CAUGHT IN EMBREE
    ERROR: Invalid operation
    ERROR MESSAGE: b'already initialized'

These can be suppressed with:

.. code-block:: python

    import logging
    logging.getLogger('pyembree').disabled = True
