from subsamplr.core.variable import Variable, DiscreteVariable, ContinuousVariable  # type: ignore
from subsamplr.core.variable_generator import VariableGenerator  # type: ignore
from numpy import ones  # type: ignore
from numpy.random import choice  # type: ignore
from mpl_toolkits import mplot3d  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


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
        """Constructor for the BinCollection class.

        Args:
            dimensions       (list): a list of Variable instances specifying
                                     the dimensions of the bin collection.
            track_exclusions (bool): a boolean value indicating whether units
                                     excluded from the collection should be tracked.
        """
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

    def assign_to_bin(self, unit, values):
        """Assign a unit to a bin and create the bin if it doesn't already exist.

        Bins are created in a nested dictionary with one level of nesting per
        dimension. The dictionary key at each level is the index of the
        partition part corresponding to all of the Bins contained inside that
        dictionary.

        Args:
            unit    (str): the name of a subsampling unit.
            values       : a collection of variable values, one per dimension.
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

    def plot(self, figsize=(6, 6), dpi=120, elev=25, azim=40, subsample=None):
        """Plot the bin collection as a 3D histogram."""

        # # TODO: optionally pass in the name of two dimensions.

        if (len(self.dimensions) < 2):
            raise ValueError("Insufficient dimensions for 3D plotting.")

        xy_pos = list()
        z_size = list()
        z_size_sub = list() # Used only if subsample is not None.
        x_dim = self.dimensions[0]
        y_dim = self.dimensions[1]

        # Determine the positions of the bars.
        for x_item in self.bins.items():
            for y_item in x_item[1].items():
                # The bar position is the index of the partition part.
                xy_pos.append((x_item[0], y_item[0]))
                z_size.append(self.count_units(d=y_item[1]))

                if subsample:
                    # Count the subsample units in the bin at this xy position.
                    count_sub = 0
                    binned_units_nested = [bin.contents for bin in self.iter(d=y_item[1])]
                    binned_units = [item for sublist in binned_units_nested for item in sublist]
                    for unit in subsample:
                        if unit in binned_units:
                            count_sub += 1
                    z_size_sub.append(count_sub)

        # Construct & draw the BinCollection plot.
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection="3d")
        plt.title(f"Complete collection of {self.count_units()} units")
        BinCollection.write_axes(ax, x_dim=x_dim, y_dim=y_dim, xy_pos=xy_pos,
            z_size=z_size, elev=elev, azim=azim)
        plt.show()

        # If a subsample was given, construct & draw the subsample plot.
        if not subsample:
            return

        fig_sub = plt.figure(figsize=figsize, dpi=dpi)
        ax_sub = plt.axes(projection="3d")
        plt.title(f"Subsample of {len(subsample)} units")
        BinCollection.write_axes(ax_sub, x_dim=x_dim, y_dim=y_dim, xy_pos=xy_pos,
            z_size=z_size_sub, elev=elev, azim=azim)
        plt.show()

    @staticmethod
    def write_axes(axes, x_dim, y_dim, xy_pos, z_size, elev, azim):
        """Write to the axes to creat a plot"""

        # Since the positions are indices, the bars have unit width.
        x_size = ones(len(xy_pos))
        y_size = ones(len(xy_pos))
        x_pos = [p[0] for p in xy_pos]
        y_pos = [p[1] for p in xy_pos]
        z_pos = [0] * len(xy_pos)

        #ax = plt.axes(projection="3d")
        axes.view_init(elev=elev, azim=azim)
        axes.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size)

        # Label the axes with the dimension names.
        axes.set_xlabel(x_dim.name)
        axes.set_ylabel(y_dim.name)
        axes.set_zlabel('Unit count')

        # Set the axis ticks and tick labels.
        xticks, xticklabels = BinCollection.ticks([i[0] for i in xy_pos], x_dim)
        yticks, yticklabels = BinCollection.ticks([i[1] for i in xy_pos], y_dim)
        axes.set_xticks(xticks)
        axes.set_yticks(yticks)
        axes.set_xticklabels(xticklabels)
        axes.set_yticklabels(yticklabels)


    @staticmethod
    def ticks(positions, dimension):
        """Get ticks and tick labels for a plot, given positions on an axis."""

        # Set the axis ticks and tick labels.
        ticks = list(set(positions))
        ticks.sort()

        if isinstance(dimension, DiscreteVariable):
            ticklabels = [dimension.partition[i].contents[0] for i in ticks]
        elif isinstance(dimension, ContinuousVariable):
            ticklabels = [dimension.partition[i].endpoints[0] for i in ticks]
        else:
            raise ValueError(f"Unexpected dimension type: {type(dimension)}")

        # Include a final tick & label at the upper bound of the axis.
        return (ticks + [ticks[-1] + 1],
            ticklabels + [max(ticklabels) + dimension.partition[-1].width()])

    @ staticmethod
    def construct(config, track_exclusions=False):
        """Static factory method.

        Constructs a BinCollection instance from configuration parameters.
        """

        dimensions = VariableGenerator.construct_variables(config)
        return BinCollection(dimensions, track_exclusions=track_exclusions)
