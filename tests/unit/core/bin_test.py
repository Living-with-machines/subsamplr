from subsamplr.core.variable import ContinuousVariable as CtsVar
from subsamplr.core.variable import DiscreteVariable as DisVar
from subsamplr.core.variable import CategoricalVariable as CatVar
from subsamplr.core.bin import Bin, BinCollection
from fractions import Fraction
import random
from numpy.random import seed as npseed  # type: ignore
import pytest  # type: ignore
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
        3. a categorical variable named 'Location'

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
        dim3.partition = ['N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW']

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

    def test_dimesion_index(self):

        # Construct an empty BinCollection.
        seed = 147
        target = self.construct_target(assign=False, seed=seed)

        assert target.dimension_index(target.dimensions[0]) == 0
        assert target.dimension_index(target.dimensions[1]) == 1
        assert target.dimension_index(target.dimensions[2]) == 2

        with pytest.raises(IndexError):
            target.dimension_index(target.dimensions[3])

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

    def test_prescribed_weights(self):

        # Construct a populated BinCollection.
        seed = 147
        size = 100
        target = self.construct_target(assign=True, size=size, seed=seed)

        # Test the Quality dimension.
        dim = target.dimensions[0]
        d = target.bins

        # Test with an invalid list of weights.
        quality_weights = [1, 1, 1, 1, 1, 2, 2, 2, 2]
        with pytest.raises(ValueError):
            target.prescribed_weights(d, dim, quality_weights, False)

        # Now test with valid weights.
        quality_weights = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        # Weights do not depend on the unit counts. They are prescribed.
        for i in range(5):
            assert target.prescribed_weights(d, dim, quality_weights, False)[i] == 1
        for i in range(5, 10):
            assert target.prescribed_weights(d, dim, quality_weights, False)[i] == 2

        for i in range(5):
            assert target.prescribed_weights(d, dim, quality_weights, True)[i] == 1/15
        for i in range(5, 10):
            assert target.prescribed_weights(d, dim, quality_weights, True)[i] == 2/15

        # Test the Year dimension.
        dim = target.dimensions[1]
        d = target.bins[0] # Descend into the first part of the Quality partition.

        # Test with an invalid list of weights (wrong length).
        year_weights = [1, 1, 1, 1, 1, 1, 1, 4, 4]
        with pytest.raises(ValueError):
            target.prescribed_weights(d, dim, year_weights, False)

        # Test with an invalid list of weights (wrong nonzero when bins are empty).
        year_weights = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4]
        with pytest.raises(ValueError):
            assert target.prescribed_weights(d, dim, year_weights, False)

        # With seed = 147, only bins with indices 2, 3, 7 & 9 are populated in this dimension slice.
        year_weights = [0, 0, 1, 1, 0, 0, 0, 4, 0, 4]
        for i in range(10):
            if i not in [2, 3, 7, 9]:
                with pytest.raises(KeyError):
                    target.prescribed_weights(d, dim, year_weights, False)[i]

        # Weights do not depend on the unit counts. They are prescribed.
        assert target.prescribed_weights(d, dim, year_weights, False)[2] == 1
        assert target.prescribed_weights(d, dim, year_weights, False)[3] == 1
        assert target.prescribed_weights(d, dim, year_weights, False)[7] == 4
        assert target.prescribed_weights(d, dim, year_weights, False)[9] == 4

        assert target.prescribed_weights(d, dim, year_weights, True)[2] == 1/10
        assert target.prescribed_weights(d, dim, year_weights, True)[3] == 1/10
        assert target.prescribed_weights(d, dim, year_weights, True)[7] == 4/10
        assert target.prescribed_weights(d, dim, year_weights, True)[9] == 4/10

        # Test the Location dimension.
        dim = target.dimensions[2]

        # Descend into the first part of the Quality partition and the last part of the Year partition.
        d = target.bins[0][9]

        # Test with an invalid list of weights (wrong length).
        location_weights = [2, 1, 2, 1, 0, 0]
        with pytest.raises(ValueError):
            target.prescribed_weights(d, dim, location_weights, False)

        # Test with an invalid list of weights (wrong nonzero when bins are empty).
        location_weights = [2, 1, 2, 1, 0, 0, 0, 0]
        with pytest.raises(ValueError):
            assert target.prescribed_weights(d, dim, location_weights, False)

        location_weights = [0, 0, 2, 1, 0, 0, 0, 0]

        # With seed = 147, only bins with indices 2, 3 & 5 are populated in this dimension slice.
        for i in range(8):
            if i not in [2, 3, 5]:
                with pytest.raises(KeyError):
                    target.prescribed_weights(d, dim, location_weights, False)[i]

        # Weights do not depend on the unit counts. They are prescribed.
        assert target.prescribed_weights(d, dim, location_weights, False)[2] == 2
        assert target.prescribed_weights(d, dim, location_weights, False)[3] == 1
        assert target.prescribed_weights(d, dim, location_weights, False)[5] == 0
       
        assert target.prescribed_weights(d, dim, location_weights, True)[2] == 2/3
        assert target.prescribed_weights(d, dim, location_weights, True)[3] == 1/3
        assert target.prescribed_weights(d, dim, location_weights, True)[5] == 0


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

        # Test select_bin with a weights argument.
        quality_weights = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        year_weights = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4]
        location_weights = [2, 1, 2, 1, 0, 0, 0, 0]
        weights = (quality_weights, year_weights, location_weights)

        npseed(seed)
        # With only 100 units in the population, these weights cannot be prescribed
        # because some bins (or bin slices) having non-zero weights are empty.
        with pytest.raises(ValueError):
            bin = target.select_bin(weights)

        # Try again with a bigger population.
        size = 5000
        target = self.construct_target(assign=True, size=size, seed=seed)
        assert len(target.units()) == size
        assert target.count_units() == size

        # With seed = 147 and size = 5000, 798 bins are created.
        assert target.count_bins() == 798

        npseed(seed)
        # With 5000 units in the population, the above weights can be prescribed.
        bin = target.select_bin(weights)
        assert isinstance(bin, Bin)
        assert bin.dimensions() == target.dimensions

        # Test select_bin with a weights argument having None entries.
        weights = (None, year_weights, None)
        npseed(seed)

        bin = target.select_bin(weights)
        assert isinstance(bin, Bin)
        assert bin.dimensions() == target.dimensions

    def test_select_units(self):

        # Construct a populated BinCollection.
        seed = 147
        size = 6000
        target = self.construct_target(assign=True, size=size, seed=seed)
        assert target.count_units() == size

        k = 100
        npseed(seed)
        result = target.select_units(k)

        assert isinstance(result, set)
        assert len(result) == k
        for unit in result:
            assert isinstance(unit, str)

        # Test select_units with prescribed bin weights.
        quality_weights = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        year_weights = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4]
        location_weights = [2, 1, 2, 1, 0, 0, 0, 0]
        weights = (quality_weights, year_weights, location_weights)

        # Check for an error if the weights are not a valid collection.
        with pytest.raises(ValueError):
            target.select_units(k, weights=year_weights)

        # With seed = 147 and size = 6000, unit selection fails because at least
        # one of the selected bins has too few units for selection without replacement.
        npseed(seed)
        with pytest.raises(ValueError):
            target.select_units(k, weights=weights)

        # With seed = 147 and size = 12000, unit selection succeeds.
        size = 12000
        npseed(seed)
        target = self.construct_target(assign=True, size=size, seed=seed)
        assert target.count_units() == size

        result = target.select_units(k, weights=weights)

        assert isinstance(result, set)
        assert len(result) == k
        for unit in result:
            assert isinstance(unit, str)

        # Test select_units with a weights argument having None entries.
        weights = (None, year_weights, None)
        npseed(seed)

        result = target.select_units(k, weights=weights)

        assert isinstance(result, set)
        assert len(result) == k
        for unit in result:
            assert isinstance(unit, str)

if __name__ == '__main__':
    unittest.main()
