from subsamplr.core.variable import ContinuousVariable as CtsVar  # type: ignore
from subsamplr.core.variable import DiscreteVariable as DisVar  # type: ignore
from fractions import Fraction

ROUNDING_DIGITS = 6


class VariableGenerator:
    """A generator of subsampling dimensions."""

    @staticmethod
    def construct_variables(config):
        """
        Constructs a collection of subsampling variables from parameters
        in a configuration file.
        """

        # Loop over the configured variables.
        vars = []
        for v in config['variables']:

            if v['class'] == 'discrete':
                # Construct a discrete variable.
                var = DisVar(v['name'], type=v['type'])
                contents_list = VariableGenerator.contents_list(
                    v['min'], v['max'], v['discretisation'], v['bin_size'], v['name'])
                var.partition = contents_list

            elif v['class'] == 'continuous':
                # Construct a continuous variable.
                var = CtsVar(v['name'], type=v['type'])
                endpoints_list = VariableGenerator.endpoints_list(
                    v['min'], v['max'], v['bin_size'], v['name'])
                var.partition = endpoints_list

            else:
                raise Exception(f"Invalid variable class: {v}")

            vars.append(var)
        return vars

    @staticmethod
    def endpoints_list(min, max, bin_size, name):
        """
        Compute a partition for the range of a continuous variable.
        Args:
            min (number): The minimum of the variable range.
            max (number): The maximum of the variable range.
            bin_size (number): The width of each bin in the partition
            name (str): The variable name.

        Either bin_size or its reciprocal must be an integer.
        The min must be an integer multiple of the bin_size.
        """

        r = max - min  # Variable range.
        bin_count = r/bin_size  # Number of bins.
        if not round(bin_count, ROUNDING_DIGITS).is_integer():
            raise ValueError(
                f"Non-integer bin count for variable {name}")
        if not round(min/bin_size, ROUNDING_DIGITS).is_integer():
            raise ValueError(
                f"Non-integer (min/bin_size) for variable {name}")
        bin_count = int(bin_count)

        # Find an integer k such that k * r is an integer.
        # Here we require that either bin_size or (1/bin_size) is an integer.
        if isinstance(bin_size, int) or bin_size.is_integer():
            k = 1
        elif round(1/bin_size, ROUNDING_DIGITS).is_integer():
            k = int(1/bin_size)
        else:
            raise ValueError(f"Invalid bin size: {bin_size}")

        # Use k to compute integer numerator & denominator for each endpoint.
        ret = []
        denom = bin_count*k
        for i in range(0, bin_count):
            a_num = int(min*denom + i*r*k)
            b_num = int(a_num + bin_size*denom)
            ret.append((Fraction(a_num, denom), Fraction(b_num, denom)))
        return ret

    @staticmethod
    def contents_list(min, max, discretisation, bin_size, name):
        """
        Compute a partition for the range of a discrete variable.
        Args:
            min (number): The infimum of the variable range.
            max (number): The suprimum of the variable range.
            discretisation (number): The discretisation width.
            bin_size (number): The width of each bin in the partition
            name (str): The variable name.

        The bin_size must be an integer multiple of the discretisation.
        The range ((max + discretisation) - min) must be an integer
        multiple of the bin_size.
        """

        if not round(bin_size/discretisation, ROUNDING_DIGITS).is_integer():
            raise ValueError(
                f"Non-integer bin_size/discretisation for variable {name}")
        if not round(((max + discretisation) - min)/bin_size, ROUNDING_DIGITS).is_integer():
            raise ValueError(
                f"Non-integer (max + discretisation - min)/bin_size for variable {name}")

        ret = []
        for i in range(min, max + 1, bin_size):
            t = tuple([i + j for j in range(0, bin_size, discretisation)])
            ret.append(t)
        return ret
