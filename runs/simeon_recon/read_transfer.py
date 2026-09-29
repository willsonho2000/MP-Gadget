"""Reader for the CLASS transfer function tables (ics_transfer_<z>.dat)."""

import re
import numpy as np


def _column_names(filename):
    """Return the column labels from the last comment line of a CLASS output file.

    The labels look like ``1:k (h/Mpc)   2:d_g   3:d_b ...``, so split on the
    ``<number>:`` markers and strip the trailing unit in parentheses.
    """
    header = ""
    with open(filename) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            header = line
    if not header:
        raise ValueError("No comment header found in %s" % filename)
    fields = re.split(r"\s*\d+:", header.lstrip("# ").rstrip())
    #The split leaves an empty string in front of the first label.
    return [f.split("(")[0].strip() for f in fields if f.strip()]


def read_transfer(filename):
    """Read a CLASS transfer function file, ics_transfer_<z>.dat.

    Returns a dict mapping column name ('k', 'd_b', 'd_cdm', 'd_tot', 'phi',
    't_tot', ...) to the corresponding array. k is in h/Mpc and the transfer
    functions are normalised to an initial curvature perturbation of 1.
    """
    data = np.loadtxt(filename)
    names = _column_names(filename)
    if len(names) != data.shape[1]:
        #CLASS still prints the ncdm labels in the header when the massive
        #neutrinos are massless or degenerate, but omits the columns.
        names = [n for n in names if "ncdm" not in n]
    if len(names) != data.shape[1]:
        raise ValueError("%s: %d header labels for %d columns"
                         % (filename, len(names), data.shape[1]))
    return dict(zip(names, data.T))


if __name__ == "__main__":
    import sys
    for fn in sys.argv[1:]:
        trans = read_transfer(fn)
        print(fn, list(trans.keys()))
