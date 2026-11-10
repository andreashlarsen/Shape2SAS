"""Base class of the models a user can select by name: the subunits and the structure factors."""

import inspect
from typing import ClassVar

from .helpfunctions import InputError


class Aliased:
    """
    a model the user can select by any of its aliases, and create from a list of values, as given on the command line

    A subclass must set the class attribute
        aliases : names the user can type for the model; the first is the canonical name
    and take its values as named arguments of __init__.
    """

    aliases: ClassVar[list[str]]
    # what the model and its values are called in error messages, set by the direct subclasses
    _kind: ClassVar[str]
    _value_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        """catch a missing or malformed class attribute when the model is defined, not when it is first used"""
        super().__init_subclass__(**kwargs)
        if "_kind" in cls.__dict__:
            return  # a base class for a kind of model, such as Subunit, has no aliases of its own

        aliases = cls.__dict__.get("aliases") # get the attribute from the class itself, not from its parents
        if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) for a in aliases):
            raise TypeError(cls._kind + " " + cls.__name__ + " must define aliases as a non-empty list of names")

    @classmethod
    def parameters(cls) -> list[str]:
        """the names of the values the model takes, in order"""
        return list(inspect.signature(cls).parameters)

    @classmethod
    def _from_values(cls, values):
        """create the model from a list of its values, in order; raises InputError, naming them, on a wrong number of values"""
        values = list(values)
        try:
            inspect.signature(cls).bind(*values)
        except TypeError:
            names = cls.parameters()
            raise InputError("%s %s takes %d %s%s (%s), but %d %s given: %s"
                             % (cls._kind, cls.aliases[0], len(names), cls._value_name, "" if len(names) == 1 else "s",
                                ", ".join(names), len(values), "was" if len(values) == 1 else "were", values)) from None
        return cls(*values)
