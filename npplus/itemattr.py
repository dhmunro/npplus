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

# See the above stackoverflow question for an introduction to this issue.
# The answers of @Kimvais and @Doug-R provide some background on the
# pros and cons of conflating dict items and instance attributes.
#
# 

from functools import partial
import sys

__all__ = ['items_are_attrs', 'ADict', 'redict']


class ItemsAreAttrs(object):
    # Mark this class as an argument to items_are_attrs decorator.
    decorator_argument = True

    # You can subclass ItemsAreAttrs and override name2key if you
    # want to customize this.
    @staticmethod
    def name2key(name):
        return name[:-1] if name.endswith('_') else name

    def get(self, name):  # __getattribute__
        if (name.startswith('__') or
            name in object.__getattribute__(self, '_IAA_class_attrs_')):
            return object.__getattribute__(self, name)
        return self[ItemsAreAttrs.name2key(name)]

    def set(self, name, value):  # __setattr__
        if name.startswith('__'):
            raise ValueError("cannot access __-prefixed item as attribute")
        self[ItemsAreAttrs.name2key(name)] = value

    def delete(self, name):  # __delattr__
        if name.startswith('__'):
            raise ValueError("cannot access __-prefixed item as attribute")
        del self[ItemsAreAttrs.name2key(name)]


def items_are_attrs(cls=None, methods=ItemsAreAttrs):
    """Class decorator to convert attribute accesses to item accesses.

    If you decorate a class with ``@items_are_attrs``, instance
    attributes will be converted to items.  That is, ``x.name`` will
    be equivalent to ``x['name']``.  The underlying class must be a
    mapping from symbol-like str keys to values for this to make
    sense.  Use `items_are_attrs` when you expect most accesses to
    items in a class instance will use quoted strings.  Not only is
    ``x.name`` easier to type than ``x['name']`` for interactive use,
    it is also easier to read.  For the same reasons, you should
    prefer ``x[name]`` to ``getattr(x, name)`` when name is not a
    quoted string.

    To permit attribute-like access to items whose keys are python
    reserved words, or methods or attributes of the underlying class,
    you may append a trailing underscore to the attribute name.  That
    is, ``x.name_`` is also equivalent to ``x['name']``.  Only a
    single trailing underscore is removed, so you have to treat any
    key which really does end in trailing underscore as if it were a
    reserved word.  Furthermore, a trailing underscore is *not*
    removed from any name beginning with leading dunder (double
    underscore), to avoid confusion with python special method and
    attribute names and name-mangling rules.

    See Also
    --------
    ADict : items_are_attrs-wrapped version of the builtin dict

    Notes
    -----
    The underscore escape convention is inspired by the PEP8
    recommendation for dealing with conflicts between variable or
    function names and python reserved words.

    The basic usage is::

        @items_are_attrs
        class MyClass(...):
            ...

    Attributes of an `items_are_attrs` class instance are not really
    instance attributes; `x.missing` will generate `KeyError`, not
    `AttributeError`.  The attribute access is merely syntactic sugar.

    You can provide an argument to `items_are_attrs` to override the
    trailing underscore escape convention or any other behavior::

        @items_are_attrs(MyAttrMethods)
        class MyClass(...):
            ...

    The `MyAttrMethods` class is a container for methods `get`, `set`,
    and `delete`, which `items_as_attrs` will copy into `MyClass` as
    its `__getattribute__`, `__setattr__`, and `__delattr__` methods.
    The `MyAttrMethods` class should be a subclass of the default,
    which is `itemattr.ItemsAreAttrs`.

    """
    if hasattr(cls, 'decorator_argument'):
        cls, methods = None, methods
    if cls is None:
        return partial(items_are_attrs, methods=methods)
    # set class attribute names for __getattribute__
    names = [n for n in dir(cls) if not n.startswith('__')]
    cls._IAA_class_attrs_ = set(names) if len(names) > 4 else names
    # methods.name would call methods.name.__get__(...), incorrectly
    # making the name function an unbound method of methods, so:
    cls.__getattribute__ = methods.__dict__['get']
    cls.__setattr__ = methods.__dict__['set']
    cls.__delattr__ = methods.__dict__['delete']
    return cls


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
    syntax.

    However, for the case of names which are reserved words or dict
    method names, ADict will modify the name by removing a single
    trailing underscore::

        x.name_   # always same as   x['name']

    The inspiration for this behavior is the PEP8 recommendation to
    avoid reserved word collisions in variable names by appending a
    trailing underscore, `class_` for `class`, `yield_` for `yield`,
    and so on.  (As PEP8 points out, if the name is under your
    control, a synonym may be a better choice.)  The convention is
    useful not only for python reserved words, but also for dict
    method names: Use `keys_` for `keys`, `items_` for `items`, etc.
    The downside is that when item keys really do end in '_', you
    must append an extra underscore.

    A trailing underscore will **not** be removed, however, for names
    beginning with double underscore, to avoid problems with python
    special method and attribute names.

    Use an ADict when you expect most accesses to items in a dict `x`
    will be quoted item names; ``x['name']`` is harder to read or to
    type in an interactive session than ``x.name``.  However, even
    when `x` is an ADict, use ``x[name]`` not ``getattr(x, name)``.

    See Also
    --------
    items_are_attrs : class decorator to provide this for any class
    redict : recursively toggle between dict and ADict

    """
    __slots__ = []

    def __getattr__(self, name):
        return self[ItemsAreAttrs.name2key(name)]

    def __setattr__(self, name, value):
        self[ItemsAreAttrs.name2key(name)] = value

    def __delattr__(self, name):
        del self[ItemsAreAttrs.name2key(name)]

    def __repr__(self):
        return "ADict(" + super(ADict, self).__repr__() + ")"


def redict(d, cls=None):
    """Recursively convert a nested dict to a nested ADict.

    Parameters
    ----------
    d : dict or ADict instance
        A dict, possibly nested, to be converted.
    cls : dict or subclass of dict, optional
        The dict-like cls to recursively convert `d` and any sub-dicts
        into.  By default, if `d` is an `ADict`, `cls` is `dict`,
        otherwise `cls` is `ADict`, so repeated calls to `redict` toggle
        between `dict` and `ADict`.

    Returns
    -------
    dnew : dict or ADict
        A copy of `d` whose class is `cls`.  Any items which are dict
        instances are similarly copied to be `cls` instances.  Non-dict
        items are not copied unless assignment makes copies.
    
    """
    if cls is None:
        cls = dict if isinstance(d, ADict) else ADict
    dnew = cls(d)
    for key, value in _iteritems(d):
        if isinstance(value, dict):
            dnew[key] = redict(value, cls)
    return dnew

_iteritems = dict.iteritems if sys.version_info[0] < 3 else dict.items
