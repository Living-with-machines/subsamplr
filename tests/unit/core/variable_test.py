from subsamplr.core.variable import ContinuousVariable as CtsVar
from subsamplr.core.variable import DiscreteVariable as DisVar
from subsamplr.core.variable import CategoricalVariable as CatVar
from fractions import Fraction
import unittest


class VariableTestCase(unittest.TestCase):

    def test_interval(self):

        variable = CtsVar("Quality")
        # The default type of a continuous variable is float.
        assert variable.type == float

        endpoints = (0, Fraction(1, 10))
        target = CtsVar.Interval(variable, endpoints)

        assert isinstance(target, CtsVar.Interval)

        # The lower bound of a continuous interval *is* contained.
        assert target.contains(0)
        assert target.contains(0.04)
        assert not target.contains(0.2)
        # The upper bound of a continuous interval is *not* contained.
        assert not target.contains(0.1)

        # The width is 1/10. 
        assert target.width() == Fraction(1, 10)

        # A discrete variable cannot be passed to the Interval constructor.
        discrete_variable = DisVar("Year")
        self.assertRaises(ValueError, CtsVar.Interval,
                          discrete_variable, (0, 0.1))

        self.assertRaises(ValueError, CtsVar.Interval,
                          variable, ("a", 0.1))
        self.assertRaises(ValueError, CtsVar.Interval,
                          variable, (0, "b"))
        self.assertRaises(ValueError, CtsVar.Interval,
                          variable, (0, 0))
        self.assertRaises(ValueError, CtsVar.Interval,
                          variable, (0.1, 0))
        self.assertRaises(ValueError, CtsVar.Interval,
                          variable, (0, 0.1, 0.2))

        # Now test with a continuous variable of type int.
        variable = CtsVar("Word count", type="int")
        assert variable.type == int

        endpoints = (1000, 2000)
        target = CtsVar.Interval(variable, endpoints)

        assert target.contains(1000)
        assert target.contains(1780)
        assert target.contains(1999.999)
        assert not target.contains(2000)
        assert not target.contains(999.99)

        # The width is 1000. 
        assert target.width() == 1000

    def test_bucket(self):

        variable = DisVar("Year")
        target = DisVar.Bucket(variable, (1855, 1856, 1857, 1858, 1859))

        assert isinstance(target, DisVar.Bucket)

        assert target.contains(1856)
        assert target.contains(1859)
        assert not target.contains(1854)
        assert not target.contains(1860)

        # The width is five years as this bucket represents the year interval [1855, 1860).        
        assert target.width() == 5
        
        # A continuous variable cannot be passed to the Bucket constructor.
        continuous_variable = CtsVar("Quality")
        self.assertRaises(ValueError, DisVar.Bucket,
                          continuous_variable, (1855, 1856, 1857, 1858, 1859))


    def test_cateogry(self):

        variable = CatVar("Year")
        target = CatVar.Category(variable, "NE")

        assert isinstance(target, CatVar.Category)

        assert target.contains("NE")
        assert not target.contains("N")
        assert not target.contains("S")
        assert not target.contains("E")

        # The width of any category is zero. 
        assert target.width() == 0
        
        # A continuous variable cannot be passed to the Category constructor.
        continuous_variable = CtsVar("Quality")
        self.assertRaises(ValueError, CatVar.Category,
                          continuous_variable, "NE")

        # A discrete variable cannot be passed to the Category constructor.
        discrete_variable = DisVar("Year")
        self.assertRaises(ValueError, CatVar.Category,
                          discrete_variable, "NE")

        # Category content must be a scalar:
        self.assertRaises(ValueError, CatVar.Category,
                          variable, ["NE"])
        self.assertRaises(ValueError, CatVar.Category,
                          variable, {"NE"})
        self.assertRaises(ValueError, CatVar.Category,
                          variable, {"NE": 1})
        self.assertRaises(ValueError, CatVar.Category,
                          variable, ("NE", "NW"))
        # But a singleton tuple is automatically converted to a scalar:
        target = CatVar.Category(variable, ("S"))


    def test_continuous_variable(self):

        target = CtsVar("Quality")
        assert isinstance(target, CtsVar)
        assert target.type == float

        # Partition the unit interval into 10 equal intervals.
        endpoints_list = [(Fraction(i, 10), Fraction(i + 1, 10))
                          for i in range(0, 10)]
        target.partition = endpoints_list

        assert len(target.partition) == 10

        endpoints = (Fraction(2, 10), Fraction(3, 10))
        # The index method works with either endpoints or an Interval.
        assert target.index(endpoints) == 2
        assert target.index(CtsVar.Interval(target, endpoints)) == 2

        for i in range(0, 10):
            assert target.index(target.partition[i]) == i

        assert target.index(0) == None

        assert target.part_containing(-0.1) == None
        assert target.part_containing(0) == 0
        assert target.part_containing(0.05) == 0
        assert target.part_containing(0.1) == 1
        assert target.part_containing(0.15) == 1
        assert target.part_containing(0.5) == 5
        assert target.part_containing(0.999) == 9
        assert target.part_containing(1) == None

        # Now test a continuous variable of type int.
        target = CtsVar("Word count", type="int")
        assert isinstance(target, CtsVar)
        assert target.type == int

        # Partition the range 0 to 150000 into 10 equal intervals.
        endpoints_list = [(i, i + 15000)
                          for i in range(0, 150000, 15000)]
        target.partition = endpoints_list

        assert len(target.partition) == 10

        endpoints = (75000, 90000)
        # The index method works with either endpoints or an Interval.
        assert target.index(endpoints) == 5
        assert target.index(CtsVar.Interval(target, endpoints)) == 5

        for i in range(0, 10):
            assert target.index(target.partition[i]) == i

        assert target.index(0) == None

        assert target.part_containing(-0.1) == None
        assert target.part_containing(0) == 0
        assert target.part_containing(14999) == 0
        assert target.part_containing(15000) == 1
        assert target.part_containing(29999) == 1
        assert target.part_containing(77461) == 5
        assert target.part_containing(135000) == 9
        assert target.part_containing(150000) == None

    def test_discrete_variable(self):

        target = DisVar("Year")
        assert isinstance(target, DisVar)

        # Partition the period 1800-1999 into decades.
        contents_list = []
        for i in range(1800, 1900, 10):
            t = tuple([i + j for j in range(0, 10)])
            contents_list.append(t)

        target.partition = contents_list

        assert len(target.partition) == 10

        for i in range(0, 10):
            assert target.index(target.partition[i]) == i
        assert target.index((1840, 1841, 1842, 1843, 1844,
                             1845, 1846, 1847, 1848, 1849)) == 4

        assert target.index(1840) == None

        assert target.part_containing(1799) == None
        assert target.part_containing(1800) == 0
        assert target.part_containing(1840) == 4
        assert target.part_containing(1842) == 4
        assert target.part_containing(1899) == 9
        assert target.part_containing(1900) == None

    def test_categorical_variable(self):

        target = CatVar("location")
        assert isinstance(target, CatVar)

        categories = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']

        # The partition in a categorical variable is the list of categories.
        target.partition = categories

        assert len(target.partition) == 8

        for i in range(0, len(categories)):
            assert target.index(target.partition[i]) == i
        assert target.index('S') == 1
        assert target.index('NE') == 4

        assert target.index('NNW') == None

        assert target.part_containing('W') == 3
        assert target.part_containing('N') == 0
        assert target.part_containing('SW') == 7
        assert target.part_containing('SSW') == None
