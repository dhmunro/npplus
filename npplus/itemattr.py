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

Provides class `ADDict` which is like `ADict` except that a missing
attribute is initialized to an empty `ADDict` instance, so you can easily
define nested dicts with ``x.a.b.c = value``.

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

__all__ = ['items_are_attrs', 'ADict', 'ADDict', 'redict']


class ItemsAreAttrs(object):
    # Mark this class as an argument to items_are_attrs decorator.
    decorator_argument = True

    # You can subclass ItemsAreAttrs and override name2key if you
    # want to customize this.
    @staticmethod
    def name2key(name):
        if not name.endswith('_') or name.startswith('__'):
            return name
        return name[:-1]

    def get(self, name):
        return self[ItemsAreAttrs.name2key(name)]

    def set(self, name, value):
        self[ItemsAreAttrs.name2key(name)] = value

    def delete(self, name):
        del self[ItemsAreAttrs.name2key(name)]

    @property
    def _ish(self):
        """Return __dict__ as an ADict.  Primarily for self._ish.attrib."""
        d = object.__getattribute__(self, '__dict__')
        if d.__class__ is not ADict:
            d = ADict(d)
            object.__setattr__(self, '__dict__', d)
        return d


def items_are_attrs(cls=None, methods=ItemsAreAttrs, ish=False):
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
    its `__getattr__`, `__setattr__`, and `__delattr__` methods.  The
    `MyAttrMethods` class should be a subclass of the default, which
    is `itemattr.ItemsAreAttrs`.

    Since an `items_are_attrs` class overrides the usual access to its
    instance attributes, referring to any instance attributes requires
    going through __dict__ explicitly (which itself works only because
    x.__dict__ does not go through x.__getattr__).  This ugliness can
    be an issue, especially in the implementaion of the class you are
    decorating -- ``self.myattr`` no longer refers to the instance
    attribute myattr, but to the item 'myattr'.  To work around that
    readability issue, you can provide the ish keyword to your
    decorator::

        @items_are_attrs(ish=1)
        class MyClass(...):
            def __init__(self, ...):
                ...
            ...

    This provides a class property `_ish`, so that for any instance,
    for example `self`, of `MyClass`, ``self._ish.myattr`` refers to
    the instance attribute myattr, not the item 'myattr'.  The first
    reference to ``self._ish`` converts __dict__ from an ordinary dict
    to an ADict.

    """
    if hasattr(cls, 'decorator_argument'):
        cls, methods = None, methods
    if cls is None: 
        return partial(items_are_attrs, methods=methods, ish=ish)
    # methods.name would call methods.name.__get__(...), incorrectly
    # making the name function an unbound method of methods, so:
    cls.__getattr__ = methods.__dict__['get']
    cls.__setattr__ = methods.__dict__['set']
    cls.__delattr__ = methods.__dict__['delete']
    if ish:
        cls._ish = methods.__dict__['_ish']
    return cls


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
    ADDict : ADict-like class for easy initialization of nested dicts
    redict : recursively toggle between dict and ADDict or ADict

    """
    __slots__ = []

    def __repr__(self):
        return "ADict(" + super(ADict, self).__repr__() + ")"


@items_are_attrs
class ADDict(dict):
    """Subclass of dict permitting access to items as if they were attributes.

    An ADDict is like an ADict, except that if you attempt to get a
    missing attribute, it will be initialized to an empty ADDict
    instance.  The only reason to use an ADDict is to be able to easily
    initialize nested dicts::

        x = ADDict()
        x.a = 'top level'
        x.b.one = 'second level item 1'
        x.b.two = 'second level item 2'
        x.b.c.yow = 'third level item'
        # instead of this:
        x = {'a': 'top level',
             'b': {'one': 'second level item 1',
                   'two': 'second level item 2',
                   'c': {'yow': 'third level item'}}}

    See Also
    --------
    redict : recursively toggle between dict and ADDict or ADict
    ADict : basic items-as-attributes dict
    items_are_attrs : class decorator to provide this for any class

    References
    ----------
    This class behaves like a stripped down version of the addict package
    at https://github.com/mewwts/addict .

    """
    __slots__ = []

    def __repr__(self):
        return "ADDict(" + super(ADDict, self).__repr__() + ")"

    def __missing__(self, key):
        self[key] = value = ADDict()
        return value


def redict(d, cls=None):
    """Recursively convert a nested dict to an ADDict or ADict instance.

    Parameters
    ----------
    d : dict or ADDict or ADict instance
        A dict, possibly nested, to be converted.
    cls : dict or subclass of dict, optional
        The dict-like cls to recursively convert `d` and any sub-dicts
        into.  By default, if `d` is an `ADDict` or `ADict`, `cls` is
        `dict`, otherwise `cls` is `ADDict`, so repeated calls to
        `redict` toggle between `dict` and `ADDict`.

    Returns
    -------
    dnew : dict or ADDict or ADict instance
        A copy of `d` whose class is `cls`.  Any items which are dict
        instances are similarly copied to be `cls` instances.  Non-dict
        items are not copied unless assignment makes copies.
    
    """
    if cls is None:
        cls = dict if d.__class__ in (ADict, ADDict) else ADDict
    dnew = cls(d)
    for key, value in _iteritems(d):
        if isinstance(value, dict):
            dnew[key] = redict(value, cls)
    return dnew

_iteritems = dict.iteritems if sys.version_info[0] < 3 else dict.items
