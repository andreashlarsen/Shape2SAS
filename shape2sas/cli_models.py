"""Subunits and structure factors from their names and values, as given on the command line.

The command line selects a model by any of its aliases (the class attribute `aliases`, whose first entry is the canonical
name) and gives its values as a list, in the order of the arguments of the model's __init__. The Python API creates the
models directly instead, e.g. Cylinder(radius=20, length=100).
"""

import inspect

from . import structure_factors, subunits
from .helpfunctions import InputError
from .structure_factors import StructureFactor
from .subunits import Subunit


def normalise_alias(name):
    """normalise a name so that 'Hollow sphere', 'hollow_sphere' and 'hollowsphere' all refer to the same thing"""
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")


def alias_registry(package):
    """map every normalised alias of every class in a package to that class"""
    registry = {}
    for _, cls in inspect.getmembers(package, inspect.isclass):
        for alias in getattr(cls, "aliases", []):
            registry[normalise_alias(alias)] = cls
    return registry


def parameter_names(cls):
    """the names of the values a model takes, in order"""
    return list(inspect.signature(cls).parameters)


def _lookup(package, name, kind):
    """look up a class of the package by any of its aliases, or list the valid names"""
    registry = alias_registry(package)
    try:
        return registry[normalise_alias(name)]
    except KeyError:
        available = sorted(set(cls.aliases[0] for cls in registry.values()))
        raise InputError("unknown %s '%s'. Available %ss: %s" % (kind, name, kind, ", ".join(available))) from None


def _construct(cls, values, kind, value_name):
    """create the model from a list of its values, in order; raises InputError, naming them, on a wrong number of values"""
    values = list(values)
    try:
        inspect.signature(cls).bind(*values)
    except TypeError:
        names = parameter_names(cls)
        raise InputError("%s %s takes %d %s%s (%s), but %d %s given: %s"
                         % (kind, cls.aliases[0], len(names), value_name, "" if len(names) == 1 else "s",
                            ", ".join(names), len(values), "was" if len(values) == 1 else "were", values)) from None
    return cls(*values)


def make_subunit(name, dimensions) -> Subunit:
    """create a subunit from any of its names and a list of its dimensions; raises InputError on invalid input"""
    return _construct(_lookup(subunits, name, "subunit"), dimensions, "subunit", "dimension")


def make_structure_factor(name, parameters) -> StructureFactor:
    """create a structure factor from any of its names and a list of its parameters; raises InputError on invalid input"""
    return _construct(_lookup(structure_factors, name, "structure factor"), parameters, "structure factor", "parameter")
