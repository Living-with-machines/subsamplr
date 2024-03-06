from subsamplr.core.bin import BinCollection  # type: ignore
from subsamplr.data.unit_generator import UnitGenerator  # type: ignore
from pandas import read_csv  # type: ignore
from io import StringIO
from numpy.random import seed as npseed  # type: ignore
import yaml
import pytest  # type: ignore


# Sample configuration for testing:
@pytest.fixture
def config():
    return \
        """
        # Subsampling dimensions
        variables:
            - {name: 'year', class: 'discrete', type: 'int', min: 1800,
                max: 1919, discretisation: 1, bin_size: 10}
            - {name: 'word_count', class: 'continuous', type: 'int',
                min: 0, max: 15000, bin_size: 1000}
            - {name: 'ocr_quality_mean', class: 'continuous', type: 'float',
                min: 0.6, max: 1, bin_size: 0.1}
        """


@pytest.fixture
def articles_100():
    return read_csv('tests/fixtures/articles_query_result_100.csv', sep=",")


@pytest.fixture
def articles_100000():
    return read_csv('tests/fixtures/articles_query_result_100000.csv', sep=",")


def test_article_binning(config, articles_100):

    # Use the BinCollection static factory method to construct an
    # instance from the configuration parameters.
    config = yaml.safe_load(StringIO(config))
    bc = BinCollection.construct(config, track_exclusions=True)

    assert isinstance(bc, BinCollection)

    # One dimension per configured variable.
    assert len(bc.dimensions) == len(config['variables'])
    for i in range(0, len(bc.dimensions)):
        assert bc.dimensions[i].name == config['variables'][i]['name']

    assert bc.count_bins() == 0
    assert bc.count_units() == 0
    assert bc.count_exclusions() == 0

    # Generate article units from the test fixture (data frame)
    # and assign to the BinCollection.
    units = UnitGenerator.generate_units(
        articles_100, unit_id="article_id", variables=bc.dimensions)
    for unit, values in units:
        bc.assign_to_bin(unit, values)

    # Of the 100 units in the table, 68 fall within the configured bin collection range.
    assert bc.count_units() == 68

    # The remaining 32 units were tracked.
    assert bc.count_exclusions() == 32

    # Each of the 100 rows is either added to the bin collection as a unit
    # or recorded as an exclusion.
    assert bc.count_units() + bc.count_exclusions() == len(articles_100.index)

    # To hold the 68 units, 18 bins were constructed.
    assert bc.count_bins() == 18


def test_article_subsampling(config, articles_100000):

    config = yaml.safe_load(StringIO(config))
    bc = BinCollection.construct(config, track_exclusions=True)

    # There are 100000 rows in the articles_100000 data frame.
    assert len(articles_100000.index) == 100000

    # Generate article units from the test fixture (data frame)
    # and assign to the bin collection.
    units = UnitGenerator.generate_units(
        articles_100000, unit_id="article_id", variables=bc.dimensions)

    generated_units = list(units)

    # Missing values in the articles_100000 data frame are skipped.
    # There are 6 missing values so only 99994 units are generated.
    assert len(generated_units) == 99994

    # Assign the generated units to the BinCollection.
    for unit, values in generated_units:
        bc.assign_to_bin(unit, values)

    # Of the 99994 units, 76361 are within the bounds of the bin collection.
    assert bc.count_units() == 76361

    # The rest are excluded from the bin collection.
    assert bc.count_exclusions() == 23633

    # Each of the 99994 units is either added to the bin collection as a unit
    # or recorded as an exclusion.
    assert bc.count_units() + bc.count_exclusions() == len(generated_units)

    # Construct a subsample of 5000 units.
    k = 5000
    seed = 147
    npseed(seed)

    sample = bc.select_units(k)

    assert isinstance(sample, set)
    assert len(sample) == k

    # The sample consists of the article identifiers.
    for unit in sample:
        assert isinstance(unit, str)
