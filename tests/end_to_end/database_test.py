from subsamplr.data.unit_generator import DbUnitGenerator  # type: ignore
from subsamplr.core.bin import BinCollection  # type: ignore
from pandas import DataFrame  # type: ignore
from io import StringIO
from numpy.random import seed as npseed  # type: ignore
import yaml  # type: ignore
import unittest


class DbUnitGeneratorTestCase(unittest.TestCase):
    """
    Test class for end-to-end tests involving a database connection.
    """

    # Sample configuration for testing:
    config = """
    # Database connection parameters
    db_dialect: 'postgresql'
    db_host: 'lwmrelationaldb.postgres.database.azure.com'
    db_port: 5432
    db_user: 'thobson'
    db_database: 'newspapers'

    # Subsampling dimensions
    variables:
        - {name: 'year', class: 'discrete', type: 'int', min: 1800,
            max: 1919, discretisation: 1, bin_size: 10}
        - {name: 'word_count', class: 'continuous', type: 'int',
            min: 0, max: 15000, bin_size: 1000}
        - {name: 'ocr_quality_mean', class: 'continuous', type: 'float',
            min: 0.6, max: 1, bin_size: 0.1}

    queries:
        article: |
            SELECT
                publication.fmp_id as nlp,
                publication.title as publication,
                publication.location as location,
                issue.issue_date as issue_date,
                CAST(EXTRACT(YEAR FROM issue.issue_date) AS INTEGER) as year,
                issue.input_sub_path as directory_path,
                article.word_count as word_count,
                article.ocr_quality_mean as ocr_quality_mean,
                article.title as article_title,
                article.fmp_id as article_fmp_id,
                issue.input_sub_path || '/' || article.fmp_id as article_id
            FROM
                publication,
                issue,
                article
            WHERE
                issue.publication_id=publication.id AND
                article.issue_id=issue.id;
    """

    def test_fetch_data(self):

        # Read the YAML config.
        config = yaml.safe_load(StringIO(self.config))

        # Limit the query to 10 results.
        limit = 10
        query = config['queries']['article'].rstrip()[:-1] + f" LIMIT {limit};"

        # Construct a database-driven subsampling unit generator.
        ug = DbUnitGenerator(config['db_dialect'], config['db_host'],
                             config['db_port'], config['db_database'],
                             config['db_user'])

        # Fetch the data.
        result = ug.fetch_data(query)

        assert isinstance(result, DataFrame)
        assert len(result.index) == limit
        assert 'article_id' in result.columns.values
        assert 'year' in result.columns.values
        assert 'word_count' in result.columns.values
        assert 'ocr_quality_mean' in result.columns.values

    def test_article_subsampling(self):

        # Read the YAML config.
        config = yaml.safe_load(StringIO(self.config))

        # Limit the query to 50,000 results.
        limit = 50000
        query = config['queries']['article'].rstrip()[:-1] + f" LIMIT {limit};"

        # Construct a database-driven subsampling unit generator.
        ug = DbUnitGenerator(config['db_dialect'], config['db_host'],
                             config['db_port'], config['db_database'],
                             config['db_user'])

        # Fetch the data.
        df = ug.fetch_data(query)

        assert len(df.index) == limit

        # Generate article units from the data and assign to a BinCollection.
        bc = BinCollection.construct(config, track_exclusions=True)
        units = DbUnitGenerator.generate_units(
            df, unit_id="article_id", variables=bc.dimensions)

        generated_units = list(units)

        # Missing values in the data frame are skipped.
        # There are 2 missing values so only 49998 units are generated.
        assert len(generated_units) == 49998

        for unit, values in generated_units:
            bc.assign_to_bin(unit, values)

        # Of the 49998 units, 34803 are within the bounds of the bin collection.
        assert bc.count_units() == 34803

        # The rest are excluded from the bin collection.
        assert bc.count_exclusions() == 15195

        # Each of the 49998 units is either added to the bin collection as a unit
        # or recorded as an exclusion.
        assert bc.count_units() + bc.count_exclusions() == len(generated_units)

        # Construct a subsample of 2000 units.
        k = 2000
        seed = 147
        npseed(seed)

        sample = bc.select_units(k)

        assert isinstance(sample, set)
        assert len(sample) == k

        # The sample consists of the article identifiers.
        for unit in sample:
            assert isinstance(unit, str)
