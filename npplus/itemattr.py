# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Access items of a mapping object as if they were attributes.

Provides class decorator `items_are_attrs` for classes that map str
keys to values.  With the decorator, items in such a class may be
accessed as attributes with the key as their name.  You can escape key
names which are python reserved words or class methods or attributes
by appending a trailing underscore when accessing the item as an
attribute.

Provides class `ADict` which wraps the builtin dict type in this way.

References
----------

See
http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
for a good discussion which mentions the following three relatively
small python packages that implement attribute-accessible dicts:

https://github.com/dsc/bunch  (combines object __dict__ with dict superclass)

https://github.com/bcj/AttrDict  (inherits from dict and other classes)

https://github.com/mewwts/addict  (focuses on recursive setattr)

--------

"""

from functools import partial

__all__ = ['items_are_attrs', 'ADict']


class ItemsAreAttrs(object):
    def get(self, name):
        return self[ItemsAreAttrs.name2key(name)]

    def set(self, name, value):
        self[ItemsAreAttrs.name2key(name)] = value

    def delete(self, name):
        del self[ItemsAreAttrs.name2key(name)]

    @staticmethod
    def name2key(name):
        if not name.endswith('_') or name.startswith('__'):
            return name
        return name[:-1]


def items_are_attrs(cls=None, methods=None):
    """Class decorator to convert attribute accesses to item accesses.

    If you decorate a class with ``@items_are_attrs``, instance
    attributes will be converted to items.  That is, ``x.name`` will
    be equivalent to ``x['name']``.  The underlying class must be
    a mapping from str keys to values for this to make sense.  Use
    `items_are_attrs` when you expect most accesses to items in a
    class instance will use quoted strings.  Not only is ``x.name``
    easier to type than ``x['name']`` for interactive use, it is
    also easier to read.  For the same reasons, you should prefer
    ``x[name]`` to ``getattr(x, name)`` when name is not a quoted
    string.

    To permit attribute-like access to items whose keys are python
    reserved words or methods or attributes of the underlying class,
    you may append a trailing underscore to the attribute name.  That
    is, ``x.name_`` is also equivalent to ``x['name']``.  Only a
    single trailing underscore is removed, so you have to treat any
    key which really does end in trailing underscore as if it were a
    reserved word.  Furthermore, a trailing underscore is *not*
    removed from any name beginning with leading double underscore, to
    avoid confusion with python special method and attribute names.

    See Also
    --------
    ADict : items_are_attrs-wrapped version of the builtin dict

    Notes
    -----
    The underscore escape convention is inspired by the PEP8
    recommendation for dealing with conflicts between variable or
    function names and python reserved words.

    You can provide a `methods` keyword to override the trailing
    underscore escape convention or any other behavior::

        @items_are_attrs(methods=MyAttrMethods)
        class MyClass(...):
            ...

    The `MyAttrMethods` class is a container for methods `get`, `set`,
    and `delete`, which `items_as_attrs` will copy into `MyClass` as
    its `__getattr__`, `__setattr__`, and `__delattr__` methods.

    """
    if methods is None:
        methods = ItemsAreAttrs
    if cls is not None:
        cls.__getattr__ = methods.__dict__['get']  # avoid __get__()
        cls.__setattr__ = methods.__dict__['set']
        cls.__delattr__ = methods.__dict__['delete']
        return cls
    else:
        return partial(items_are_attrs, methods=methods)


@items_are_attrs
class ADict(dict):
    """Subclass of dict permitting access to items as if they were attributes.

    For an ADict instance `x`::

        value = x.name  # same as   value = x['name']
        x.name = value  # same as   x['name'] = value
        del x.name      # same as   del x['name']

    For these equivalences to work, obviously `x.name` must be legal
    python syntax.  `x` may still contain keys which are reserved
    names, include punctuation characters or are not strings at all,
    but such items are accessible only using the usual ``x[...]``
    syntax.  For the case of names which are reserved words or dict
    method names, however, ADict will modify the name by removing a
    single trailing underscore::

        x.name_   # always same as   x['name']

    The inspiration for this behavior is the PEP8 recommendation to
    avoid reserved word collisions in variable names by appending a
    trailing underscore, `class_` for `class`, `yield_` for `yield`,
    and so on.  (As PEP8 points out, if the name is under your
    control, a synonym may be a better choice.)  The convention is
    useful not only for python reserved words, but also for dict
    method names: Use `keys_` for `keys`, `items_` for `items`, etc.
    The downside is that if item keys really do end in '_', you
    must append an extra underscore.

    A trailing underscore will *not* be removed, however, for names
    beginning with double underscore, to avoid problems with python
    special method and attribute names.

    Use an ADict when you expect most accesses to items in a dict `x`
    will be quoted item names; ``x['name']`` is harder to read or to
    type in an interactive session than ``x.name``.  However, even
    when `x` is an ADict, use ``x[name]`` not ``getattr(x, name)``.

    See Also
    --------
    items_are_attrs : class decorator to provide this for any class

    """
    __slots__ = []
    def __repr__(self):
        return "ADict(" + super(ADict, self).__repr__() + ")"
