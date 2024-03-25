from subsamplr.core.variable import ContinuousVariable as CtsVar  # type: ignore
from subsamplr.core.variable import DiscreteVariable as DisVar  # type: ignore
from subsamplr.core.variable import CategoricalVariable as CatVar  # type: ignore
from subsamplr.core.variable_generator import VariableGenerator  # type: ignore
from fractions import Fraction
from io import StringIO
import yaml  # type: ignore
import unittest


class VariableGeneratorTestCase(unittest.TestCase):

    def test_endpoints_list(self):

        result = VariableGenerator.endpoints_list(0, 1, 0.1, "Cts var")

        assert len(result) == (1 - 0)/0.1

        assert result[0][0] == 0
        assert result[0][1] == Fraction(1, 10)
        assert result[1][0] == Fraction(1, 10)
        assert result[1][1] == Fraction(2, 10)
        assert result[2][0] == Fraction(2, 10)
        assert result[8][0] == Fraction(8, 10)
        assert result[8][1] == Fraction(9, 10)
        assert result[9][0] == Fraction(9, 10)
        assert result[9][1] == 1

        result = VariableGenerator.endpoints_list(1, 2, 0.2, "Cts var")

        assert len(result) == (2 - 1)/0.2

        assert result[0][0] == Fraction(5, 5)
        assert result[0][1] == Fraction(6, 5)
        assert result[1][0] == Fraction(6, 5)
        assert result[1][1] == Fraction(7, 5)
        assert result[2][0] == Fraction(7, 5)
        assert result[2][1] == Fraction(8, 5)
        assert result[3][0] == Fraction(8, 5)
        assert result[3][1] == Fraction(9, 5)
        assert result[4][0] == Fraction(9, 5)
        assert result[4][1] == Fraction(10, 5)

        result = VariableGenerator.endpoints_list(0.6, 0.9, 0.1, "Cts var")

        assert len(result) == int((0.9 - 0.6)/0.1)

        assert result[0][0] == Fraction(6, 10)
        assert result[0][1] == Fraction(7, 10)
        assert result[1][0] == Fraction(7, 10)
        assert result[1][1] == Fraction(8, 10)
        assert result[2][0] == Fraction(8, 10)
        assert result[2][1] == Fraction(9, 10)

        result = VariableGenerator.endpoints_list(-1, 2.5, 0.5, "Cts var")

        assert len(result) == (2.5 + 1)/0.5

        for x in result:
            assert isinstance(x[0], Fraction)
            assert isinstance(x[1], Fraction)

        assert result[0][0] == -1
        assert result[0][1] == Fraction(-1, 2)
        assert result[1][0] == Fraction(-1, 2)
        assert result[1][1] == 0
        assert result[2][0] == 0
        assert result[2][1] == Fraction(1, 2)
        assert result[3][0] == Fraction(1, 2)
        assert result[3][1] == 1
        assert result[4][0] == 1
        assert result[4][1] == Fraction(3, 2)
        assert result[5][0] == Fraction(3, 2)
        assert result[5][1] == 2
        assert result[6][0] == 2
        assert result[6][1] == Fraction(5, 2)

    def test_contents_list(self):

        result = VariableGenerator.contents_list(
            1800, 1919, 1, 10, "Discrete var")

        assert len(result) == ((1919 + 1) - 1800)/10

        assert result[0] == (1800, 1801, 1802, 1803, 1804,
                             1805, 1806, 1807, 1808, 1809)
        assert result[1] == (1810, 1811, 1812, 1813, 1814,
                             1815, 1816, 1817, 1818, 1819)
        assert result[len(result) - 1] == (1910, 1911, 1912,
                                           1913, 1914, 1915, 1916, 1917, 1918, 1919)

        result = VariableGenerator.contents_list(
            1823, 1906, 1, 4, "Discrete var")

        assert len(result) == ((1906 + 1) - 1823)/4

        assert result[0] == (1823, 1824, 1825, 1826)
        assert result[1] == (1827, 1828, 1829, 1830)
        assert result[len(result) - 1] == (1903, 1904, 1905, 1906)

        result = VariableGenerator.contents_list(
            12, 45, 3, 12, "Discrete var")

        assert len(result) == ((45 + 3) - 12)/12

        assert result[0] == (12, 15, 18, 21)
        assert result[1] == (24, 27, 30, 33)
        assert result[2] == (36, 39, 42, 45)

    def test_construct_variables(self):

        # Sample configuration for testing:
        config = """
        # Subsampling dimensions
        variables:
            - {name: 'Year', class: 'discrete', type: 'int', min: 1800,
                max: 1919, discretisation: 1, bin_size: 10}
            - {name: 'Mean OCR quality', class: 'continuous', type: 'float',
                min: 0.6, max: 1, bin_size: 0.1}
            - {name: 'Location', class: 'categorical', type: 'str',
                categories: ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']}
        """

        config = yaml.safe_load(StringIO(config))
        result = VariableGenerator.construct_variables(config)

        assert len(result) == 3
        assert isinstance(result[0], DisVar)

        dis_var = result[0]
        assert dis_var.name == 'Year'
        assert len(dis_var.partition) == 12
        assert isinstance(dis_var.partition[3], DisVar.Bucket)
        assert dis_var.partition[3].variable == dis_var
        assert dis_var.partition[3].contents == (
            1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839)

        assert isinstance(result[1], CtsVar)

        cat_var = result[1]
        assert cat_var.name == 'Mean OCR quality'
        assert len(cat_var.partition) == 4
        assert isinstance(cat_var.partition[3], CtsVar.Interval)
        assert cat_var.partition[3].variable == cat_var
        assert cat_var.partition[3].endpoints == (Fraction(9, 10), 1)

        assert isinstance(result[2], CatVar)

        cat_var = result[2]
        assert cat_var.name == 'Location'
        assert len(cat_var.partition) == 8
        assert isinstance(cat_var.partition[3], CatVar.Category)
        assert cat_var.partition[3].variable == cat_var
        assert cat_var.partition[3].content == 'W'
