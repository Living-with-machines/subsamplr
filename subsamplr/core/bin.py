from subsamplr.core.variable import Variable  # type: ignore
from subsamplr.core.variable_generator import VariableGenerator  # type: ignore
from numpy.random import choice  # type: ignore


class Bin:
    """A histogram bin."""

    def __init__(self, parts):
        """Constructor for the Bin class.

        Args:
            parts (list): Partition elements defining the bin.
        """
        self.contents = set()
        self.parts = parts

    def count(self):
        """Count the number of units in this bin."""
        return len(self.contents)

    def contains(self, unit):
        """Check whether this bin contains the given unit."""
        return unit in self.contents

    def assign(self, unit):
        """Assign a unit to the bin (if it isn't already assigned)."""
        if self.contains(unit):
            # TODO:
            # logging.warn(f"Bin already contains unit {unit}")
            return
        self.contents.add(unit)

    def dimensions(self):
        """Return the list of variables defining the dimensions of the bin."""
        return [part.variable for part in self.parts]


class BinCollection:
    """A collection of histogram bins."""

    def __init__(self, dimensions, track_exclusions=True):
        """Constructor for the BinCollection class."""
        # Store the bins in a nested dictionary for efficient selection & assignment.
        self.bins = {}
        for dim in dimensions:
            if not isinstance(dim, Variable):
                raise ValueError(
                    "All dimensions elements must be of type Variable.")
        self.dimensions = dimensions
        self.track_exclusions = track_exclusions

        # If track_exclusions is True, create a dictionary to store exclusions.
        if track_exclusions:
            self.exclusions = dict()  # type: dict

        # self.calls_to_assign = 0

    def assign_to_bin(self, unit, values):
        """Assign a unit to a bin and create the bin if it doesn't already exist.

        Bins are created in a nested dictionary with one level of nesting per
        dimension. The dictionary key at each level is the index of the
        partition part corresponding to all of the Bins contained inside that
        dictionary.

        Args:
            unit    (str): the name of a subsampling unit
            values       : a collection of variable values, one per dimension
        """
        # TODO: log debug message if out of range:
        # logging.debug(f"unit {unit} out of range.")

        if len(values) != len(self.dimensions):
            raise ValueError(
                "Bin assignment requires one value per dimension.")

        # Get the indices of the partition parts containing the unit values.
        dim_value_pairs = zip(self.dimensions, values)
        part_indices = [dim.part_containing(value)
                        for dim, value in dim_value_pairs]

        # If any of the part_indices is None, the unit values do not fall within
        # the range of the bin collection so the unit is excluded. Keep track
        # of such exclusions only if the track_exclusions attribute is True.
        if None in part_indices:
            if self.track_exclusions:
                self.exclusions[unit] = values
            return

        # Initialise the dictionary $d$ to equal the `bins` instance variable.
        d = self.bins

        for i, dim in zip(part_indices, self.dimensions):

            # If the part index is not a key in the dictionary $d$ ...
            if not i in d.keys():

                if dim == self.dimensions[-1]:
                    # If this is the last dimension, create a new Bin in the
                    # collection and then add the unit.
                    d[i] = Bin([v.partition[i] for v in self.dimensions])
                    d[i].assign(unit)
                else:
                    # If this isn't the last dimension, add it to the keys with
                    # an empty dict as the value.
                    d[i] = dict()

            # If the part index *is* a key in the dictionary $d$ ...
            if dim == self.dimensions[-1]:
                # If this is the last dimension, add the unit to the bin.
                d[i].assign(unit)
            # Otherwise, descend down one level into the nested dictionary.
            d = d[i]

    def select_bin(self):
        """Select a bin at random, weighted by the size of the bin."""
        # TODO. Consider optimising by selecting many bins at once.

        d = self.bins
        for _ in self.dimensions:
            # Get the weights for the parts in this dimension.
            weights = self.weight_of_parts(d, normalised=True)

            # Pick one part at random, according to the weights.
            i = choice(list(weights.keys()), p=list(weights.values()))

            # If the value of the selected part is a Bin, return it.
            if isinstance(d[i], Bin):
                return d[i]

            # Otherwise descend to the next level in the nested dictionary.
            d = d[i]
        raise RuntimeError("Bin selection failed.")

    def select_units(self, k):
        """Select units at random, weighting by bin sizes.

        Args:
            k       (int): The number of items to select.
            seed    (int): A seed for RNG initialisation.

        Return:
            A set of units (strings).

        Throws:
            ValueError if the number of selections from a particular bin turns
            out to be greater than the bin size.
        """
        selection = set()
        bins = [self.select_bin() for _ in range(k)]
        while len(bins) != 0:
            # Consider the first bin selected.
            bin = bins[0]
            # Count how many times this bin appears in the bins selection.
            size = len([b for b in bins if b == bin])
            # Sample without replacement.
            selection.update(
                choice(list(bin.contents), size=size, replace=False))
            bins = [b for b in bins if b != bin]
        return selection

    def weight_of_parts(self, d, normalised):
        """Find the weights of partition parts in a particular dimension.

        Count the number of units assigned below a particular set of partition
        parts.

        Args:
            d          (dict): A dictionary within the nested bins attribute.
            normalised (bool): If True, the weights are normalised as a
                               probability distribution.

        Returns:
            A dictionary keyed by the populated partition indices for the given
            dimension, with unit counts as the corresponding values.
        """

        # The parts of interest are the keys in the given dictionary.
        ret = dict()
        for key in d.keys():
            if isinstance(d[key], dict):
                ret[key] = self.count_units(d=d[key])
            else:
                ret[key] = d[key].count()
        if not normalised:
            return ret
        total_weight = sum(ret.values())
        for key in ret.keys():
            ret[key] = ret[key]/total_weight
        return ret

    def iter(self, d=None):
        """Gernerator for iterating over bins in the collection.

        Args:
            d (dict): A dictionary within the nested bins attribute under which
                      to look for bins.
        """

        # TODO. Is order important here? Hopefully not.

        if not d:
            d = self.bins
        # Iterate over all keys of the dict argument.
        for key in d.keys():
            # Check if value is of dict type.
            if isinstance(d[key], dict):
                # If value is a dict then recursively call this generator
                # to iterate over all values in the subdictionary.
                yield from self.iter(d=d[key])
            else:
                # If value is not dict type then yield the value.
                yield d[key]

    def units(self, d=None):
        """Get all of the units assigned to bins in (part of) this collection.

        Args:
            d (dict): A dictionary within the nested bins attribute under which
                      to look for units.

        Returns:
            A set of unit identifiers (strings).
        """

        units = set()
        for bin in self.iter(d=d):
            units.update(bin.contents)
        return units

    def count_bins(self, d=None):
        """Count the number of bins in the collection.

        Args:
            d (dict): A dictionary within the nested bins attribute from which
                      to start counting.
        """
        return sum(1 for dummy in self.iter(d))

    def count_units(self, d=None):
        """Count the number of units in the bins in this collection.

        Args:
            d (dict): A dictionary within the nested bins attribute from which
                      to start counting.
        """
        return len(self.units(d))

    def count_exclusions(self):
        """Count the number of exclusions in this collection."""
        return len(self.exclusions)

    @staticmethod
    def construct(config, track_exclusions):
        """Static factory method.

        Constructs a BinCollection instance from configuration parameters.
        """

        dimensions = VariableGenerator.construct_variables(config)
        return BinCollection(dimensions, track_exclusions=track_exclusions)
