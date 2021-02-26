# Poenta
Poenta is an open source library for the automated design of quantum optical devices.

Warning: this library is still under heavy development.

Look how easy it is to use:

    from poenta.circuit import Circuit
    from poenta.nputils import single_photon,vaccum
    import tensorflow as tf

    cutoff = 50
    state_in = vaccum(1,cutoff)
    target_out = single_photon(1,cutoff)

    device = Circuit(num_layers=20, num_modes=1, num_seed = 655, dtype=tf.complex128)

    tuple_in_out = (state_in,target_out),
    device.set_input_output_pairs(*tuple_in_out)
    device.optimize(steps=50)


Features
--------

- Be awesome
- Make things faster

Installation
------------

Install $project by running:

    install project

Contribute
----------

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

Support
-------

If you are having issues, please let us know.
We have a mailing list located at: project@google-groups.com

License
-------

The project is licensed under the GNU General Public License.
