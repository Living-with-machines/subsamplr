from numbers import Number
from fractions import Fraction


class Variable:
    """A variable used for subsampling."""

    def __init__(self, name, type):
        """Constructor for the Variable class.

        Args:
            name       (str): The name of the variable.
            type       (str): The variable type.

        The type argument must be a Python built-in type, e.g. int, float.
        """
        self.name = name
        try:
            self.type = getattr(__import__("builtins"), type)
        except AttributeError:
            raise ValueError(f"Invalid Variable type: {type}")

    @property
    def partition(self):
        return self.__partition

    @partition.setter
    def partition(self, partition):
        """Setter for the partition property.

        Args:
            partition (list): A list from which a partition of this variable's range can be constructed.
        """
        if not isinstance(partition, list):
            raise ValueError("Variable partition must be a list.")
        self.__partition = [self.construct_part(self, e) for e in partition]

    # Abstract method.
    def construct_part(self, variable, arg):
        raise NotImplementedError("Abstract method.")

    # Abstract method.
    def index(self, part):
        """Get the index of a given part of the range of this
        variable, or None if the part is not contained in this variable.

        Args:
            part    : An element of the variable's partition.
        """
        raise NotImplementedError("Abstract method.")

    def part_containing(self, value):
        """Get the index of the partition element that contains the given value.

        Args:
            value   : A value in the range of the variable.

        Returns:
            The index of the partition element that contains the given value, or
            None if there is no such partition element.
        """
        for part in self.partition:
            if part.contains(value):
                return self.index(part)


class ContinuousVariable(Variable):
    """Real-valued numerical variable."""

    def __init__(self, name, type="float"):
        """ Constructor for the ContinuousVariable class.

        Args:
            name             (str): The name of the variable.
            type             (str): The variable type.
        """
        self.part_class = ContinuousVariable.Interval
        super().__init__(name=name, type=type)

    def construct_part(self, variable, endpoints):
        """Construct a part of the variable range partition."""
        return ContinuousVariable.Interval(variable, endpoints)

    def index(self, part):
        """Get the index of a given part of the range of this
        variable, or None if the part is not contained in this variable.

        Args:
            part    : An Interval or a pair of endpoints.
        """
        # Check the endpoints.
        if isinstance(part, ContinuousVariable.Interval):
            part = part.endpoints
        endpoints_list = [interval.endpoints for interval in self.partition]
        if not part in endpoints_list:
            return None
        return endpoints_list.index(part)

    class Interval:
        """Part of the range partition in a continuous variable."""

        def __init__(self, variable, endpoints):
            """Interval constructor

            Args:
                variable (ContinuousVariable) : A continuous variable whose
                    range partition contains this interval.
                endpoints (tuple) : A pair of numbers specifying the interval endpoints
                    which is interpreted as closed at the lower end and open at
                    the higher end.
            """
            if not isinstance(variable, ContinuousVariable):
                raise ValueError(
                    "Interval constructor requires a ContinuousVariable.")
            self.variable = variable

            if not isinstance(endpoints, tuple) or len(endpoints) != 2:
                raise ValueError("Interval endpoints must be a pair.")
            for endpoint in endpoints:
                if not (isinstance(endpoint, int) or isinstance(endpoint, Fraction)):
                    raise ValueError(
                        f"Interval endpoints must be ints or Fractions: {endpoint}")
            if endpoints[0] >= endpoints[1]:
                msg = "Interval endpoints must be in order"
                raise ValueError(
                    f"{msg} but {endpoints[0]} >= {endpoints[1]}.")
            self.endpoints = endpoints

        def __str__(self):
            return f"{self.variable.name}: {self.endpoints}"

        # Override the contains method in the continuous variable case.
        def contains(self, value):
            """
            Determine whether this interval contains a given numerical value.

            Args:
                value   : A numerical value.
            Returns:
                bool    : True iff this interval contains the value.
            """
            if not isinstance(value, Number):
                return False
            return value >= self.endpoints[0] and value < self.endpoints[1]

        def width(self):
            """Get the width of the interval"""

            return self.endpoints[1] - self.endpoints[0]


class DiscreteVariable(Variable):
    """Discrete-valued variable."""

    def __init__(self, name, type="int"):
        """ Constructor for the DiscreteVariable class.
        Args:
            name             (str): The name of the variable.
            type             (str): The variable type.
        """
        self.part_class = DiscreteVariable.Bucket
        super().__init__(name=name, type=type)

    def construct_part(self, variable, arg):
        return DiscreteVariable.Bucket(variable, arg)

    # Override the index method in the DiscreteVariable case.
    def index(self, part):
        """Get the index of a given part of the range of this
        variable, or None if the part is not contained in this variable.

        Args:
            part    : An element of the variable's partition.
        """
        # Check the contents.
        if isinstance(part, DiscreteVariable.Bucket):
            part = part.contents
        contents_list = [bucket.contents for bucket in self.partition]
        if not part in contents_list:
            return None
        return contents_list.index(part)

    class Bucket:
        """Part of the range partition in a discrete variable."""

        def __init__(self, variable, contents):
            """Bucket constructor

            Args:
                variable (DiscreteVariable) : A discrete variable whose
                    range partition contains this bucket.
                contents (tuple) : A tuple of bucket contents.
            """
            if not isinstance(variable, DiscreteVariable):
                raise ValueError(
                    "Bucket constructor requires a DiscreteVariable.")
            self.variable = variable
            if not isinstance(contents, tuple):
                raise ValueError("Bucket contents must be a tuple.")
            self.contents = contents

        def __str__(self):
            return f"{self.variable.name}: {self.contents[0]}, ..., {self.contents[-1]}"

        def contains(self, value):
            return value in self.contents

        def width(self):
            """Get the width of the bucket"""
            c = self.contents
            ret = max(c) - min(c)

            # If there is a constant difference between consecutive
            # elements in the contents, add that difference to the width.
            if len(set([t - s for s, t in zip(c, c[1:])])):
                ret += c[1] - c[0]

            return ret


class CategoricalVariable(Variable):
    """Categorical variable."""

    def __init__(self, name, type="str"):
        """ Constructor for the CategoricalVariable class.
        Args:
            name             (str): The name of the variable.
            type             (str): The variable type.
        """
        self.part_class = CategoricalVariable.Category
        super().__init__(name=name, type=type)

    def construct_part(self, variable, arg):
        return CategoricalVariable.Category(variable, arg)

    # Override the index method in the CategoricalVariable case.
    def index(self, part):
        """Get the index of a given part of the range of this
        variable, or None if the part is not contained in this variable.

        Args:
            part    : An element of the variable's partition.
        """
        # Check the content.
        if isinstance(part, CategoricalVariable.Category):
            part = part.content
        content_list = [category.content for category in self.partition]
        if not part in content_list:
            return None
        return content_list.index(part)

    class Category:
        """One of the categories in a categorical variable."""

        def __init__(self, variable, content):
            """Category constructor

            Args:
                variable (CategoricalVariable) : A categorical variable whose
                    categories include this category.
                content     : A scalar value representing the category content.
            """
            if not isinstance(variable, CategoricalVariable):
                raise ValueError(
                    "Category constructor requires a CategoricalVariable.")
            self.variable = variable

            if isinstance(content, tuple) or isinstance(content, list) \
                or isinstance(content, set) or isinstance(content, dict):
                raise ValueError("Category content must be a scalar.")

            self.content = content

        def __str__(self):
            return f"{self.variable.name}: {self.content}"

        def contains(self, value):
            return value == self.content

        def width(self):
            """Get the width of the category"""
            # Every category is a singleton so has width zero.
            return 0
