from subsamplr.core.variable import ContinuousVariable as CtsVar
from subsamplr.core.variable import DiscreteVariable as DisVar
from subsamplr.core.variable import CategoricalVariable as CatVar
from subsamplr.core.bin import Bin, BinCollection
from fractions import Fraction
import random
from numpy.random import seed as npseed  # type: ignore
import unittest


class BinTestCase(unittest.TestCase):

    def test_bin(self):

        var1 = CtsVar("Quality")
        part1 = CtsVar.Interval(var1, (Fraction(4, 10), Fraction(5, 10)))

        var2 = DisVar("Year")
        part2 = DisVar.Bucket(var2, (1855, 1856, 1857, 1858, 1859))

        var3 = CatVar("Location")
        part3 = CatVar.Category(var3, "N")

        parts = [part1, part2, part3]
        target = Bin(parts)

        # Test the dimensions method.
        assert target.dimensions() == [var1, var2, var3]

        # Test the assign & count methods.
        unit = "XXX"
        assert not target.contains(unit)
        assert target.count() == 0

        target.assign(unit)
        assert target.contains(unit)
        assert target.count() == 1

        # Units cannot be assigned more than once.
        target.assign(unit)
        assert target.contains(unit)
        assert target.count() == 1

        other_unit = "XYZ"
        target.assign(other_unit)
        assert target.contains(other_unit)
        assert target.count() == 2


class BinCollectionTest(unittest.TestCase):

    def construct_target(self, assign=False, size=1000, seed=147):
        """Helper method.

        Constructs three variables:
        1. a continuous variable named 'Quality'
        2. a discrete variable named 'Year'
        3. a categorical variable name 'Location'

        Defines a partition of the range of each variable.

        Constructs a BinCollection along the three dimensions.

        If assign is True, generates units with randomly chosen variable values
        and assigns the units to the BinCollection.
        """

        # Construct a continuous Quality dimension.
        dim1 = CtsVar("Quality")
        endpoints_list = [(Fraction(i, 10), Fraction(i + 1, 10))
                          for i in range(0, 10)]
        dim1.partition = endpoints_list

        # Construct a discrete Year dimension.
        dim2 = DisVar("Year")
        contents_list = []
        for i in range(1800, 1900, 10):
            t = tuple([i + j for j in range(0, 10)])
            contents_list.append(t)
        dim2.partition = contents_list

        # Construct a categorical Location dimension.
        dim3 = CatVar("Location")
        dim3.partition = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']

        dimensions = [dim1, dim2, dim3]
        target = BinCollection(dimensions)
        if not assign:
            return target

        # Generate some units and variable values.
        random.seed(seed)

        units = [f"U{i}" for i in range(0, size)]
        qualities = [q/100 for q in random.choices(range(0, 100), k=size)]
        years = random.choices(range(1800, 1900), k=size)
        locations = random.choices(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], k=size)

        all_values = zip(qualities, years, locations)

        for unit, values in zip(units, all_values):
            target.assign_to_bin(unit, values)
        return target

    def test_assign_to_bin(self):

        # Construct an empty BinCollection.
        seed = 147
        target = self.construct_target(assign=False, seed=seed)

        unit = "XXX"  # String identifier for a subsampling unit.

        assert len(target.bins) == 0

        values = (0.65, 1882, 'NE')
        target.assign_to_bin(unit, values)

        assert len(target.bins) == 1
        assert target.count_bins() == 1

        assert len(target.units()) == 1
        assert unit in target.units()

        # Check that a unit can only be assigned once.
        target.assign_to_bin(unit, values)

        assert len(target.bins) == 1

        assert len(target.units()) == 1
        assert unit in target.units()

        # The bins attribute is a nested dictionary
        dim1 = target.dimensions[0]
        dim2 = target.dimensions[1]
        assert dim1.part_containing(0.65) in target.bins.keys()
        assert dim2.part_containing(
            1882) in target.bins[dim1.part_containing(0.65)].keys()

        # Assign a second unit to the same bin
        second_unit = "YYY"  # String identifier for a subsampling unit.
        assert unit in target.units()
        assert not second_unit in target.units()

        # Use the same values as the first unit.
        target.assign_to_bin(second_unit, values)

        assert len(target.bins) == 1
        assert target.count_bins() == 1

        assert len(target.units()) == 2
        assert unit in target.units()
        assert second_unit in target.units()

        # Assign a third unit to a different bin
        third_unit = "ZZZ"  # String identifier for a subsampling unit.
        assert unit in target.units()
        assert second_unit in target.units()

        # Use different values this time.
        other_values = (0.22, 1876, 'S')
        target.assign_to_bin(third_unit, other_values)

        assert len(target.bins) == 2
        assert target.count_bins() == 2

        assert len(target.units()) == 3
        assert unit in target.units()
        assert second_unit in target.units()
        assert third_unit in target.units()

    def test_weight_of_parts(self):

        # Construct a populated BinCollection.
        seed = 147
        size = 100
        target = self.construct_target(assign=True, size=size, seed=seed)
        assert len(target.units()) == size

        assert target.count_units() == size

        # With seed = 147, 96 bins are created.
        assert target.count_bins() == 96

        # With seed = 147, check the weights in the first dimension.
        assert target.weight_of_parts(target.bins, False)[0] == 8
        assert target.weight_of_parts(target.bins, False)[1] == 10
        assert target.weight_of_parts(target.bins, False)[2] == 15
        assert target.weight_of_parts(target.bins, False)[3] == 10
        assert target.weight_of_parts(target.bins, False)[4] == 11
        assert target.weight_of_parts(target.bins, False)[5] == 11
        assert target.weight_of_parts(target.bins, False)[6] == 9
        assert target.weight_of_parts(target.bins, False)[7] == 15
        assert target.weight_of_parts(target.bins, False)[8] == 8
        assert target.weight_of_parts(target.bins, False)[9] == 3

        # With seed = 147, check the weights in (some of) the second dimension.
        assert target.weight_of_parts(target.bins[0], False)[2] == 3
        assert target.weight_of_parts(target.bins[0], False)[3] == 1
        assert target.weight_of_parts(target.bins[0], False)[7] == 1
        assert target.weight_of_parts(target.bins[0], False)[9] == 3

    def test_select_bin(self):

        # Construct a populated BinCollection.
        seed = 147
        size = 100
        target = self.construct_target(assign=True, size=size, seed=seed)
        assert len(target.units()) == size

        assert target.count_units() == size

        # With seed = 147, 96 bins are created.
        assert target.count_bins() == 96

        npseed(seed)
        bin = target.select_bin()
        assert isinstance(bin, Bin)
        assert bin.dimensions() == target.dimensions

    def test_select_units(self):

        # Construct a populated BinCollection.
        seed = 147
        size = 10000
        target = self.construct_target(assign=True, size=size, seed=seed)
        assert target.count_units() == size

        # for bin in target.iter():
        #     print(bin)
        #     print(bin.count())

        k = 100
        npseed(seed)
        result = target.select_units(k)

        assert isinstance(result, set)
        assert len(result) == k
        for unit in result:
            assert isinstance(unit, str)


if __name__ == '__main__':
    unittest.main()
