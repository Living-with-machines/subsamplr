from subsamplr.data.unit_generator import UnitGenerator  # type: ignore
from subsamplr.core.variable import ContinuousVariable as CtsVar  # type: ignore
from subsamplr.core.variable import DiscreteVariable as DisVar  # type: ignore
import pytest
from pandas import read_csv  # type: ignore
from io import StringIO


@pytest.fixture
def articles_query_result():

    data = StringIO(
        """nlp,publication,location,issue_date,year,directory_path,word_count,ocr_quality_mean,article_title,article_fmp_id,article_id
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,7155,0.4612,,art0038,0002325/1864/0609/art0038
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,214,0.5899,1101-CDEN.,art0032,0002325/1864/0609/art0032
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,2058,0.6705,IFACTURERS and IM. SOUTHAMPTON,art0018,0002325/1864/0609/art0018
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,11,0.4782,,art0037,0002325/1864/0609/art0037
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,7589,0.6361,ILE AND,art0007,0002325/1864/0609/art0007
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,1268,0.656,OLAMTS from 143 to 421 per Dozen.,art0047,0002325/1864/0609/art0047
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,528,0.9316,AMERICAN AFFAIRS.,art0030,0002325/1864/0609/art0030
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,249,0.7291,,art0027,0002325/1864/0609/art0027
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,1156,0.7865,,art0012,0002325/1864/0609/art0012
        0002325,"The Poole and South-Western Herald, etc.","Poole, Dorset, England",1864-06-09,1864,0002325/1864/0609,1577,0.4233,LVKDIG?OS.,art0033,0002325/1864/0609/art0033
        """)
    return read_csv(data, sep=",")


def test_generate_units(articles_query_result):

    # Construct variables for the year, word count and OCR quality dimensions.
    year = DisVar("year")
    word_count = CtsVar("word_count", type="int")
    ocr_quality = CtsVar("ocr_quality_mean", type="float")
    # TODO: test for a warning when ocr_quality has type "int".
    variables = [year, word_count, ocr_quality]

    result = UnitGenerator.generate_units(
        articles_query_result, unit_id="article_id", variables=variables)
    results = [unit for unit in result]

    assert len(results) == 10

    # Each element of the result is a 2-tuple (unit ID, values),
    # and values is a 3-tuple (year, word_count, ocr_quality_mean).
    for x in results:
        assert len(x) == 2
        assert isinstance(x[0], str)
        assert isinstance(x[1], tuple)
        assert len(x[1]) == 3

    # The unit identifier is the article_id column.
    # e.g. "0002325/1864/0609/art0038"
    assert results[0][0] == articles_query_result.iloc[0]['article_id']

    # The values are (year, word_count, ocr_quality_mean).
    assert results[8][1] == (1864, 1156, 0.7865)

    for i in range(0, len(results)):
        row = articles_query_result.iloc[i]

        # Check the unit identifiers.
        assert results[i][0] == row['article_id']

        # Check the unit values.
        assert results[i][1] == (
            row['year'], row['word_count'], row['ocr_quality_mean'])

        # Check the types of the unit values.
        for j in range(0, len(variables)):
            assert isinstance(results[i][1][j], variables[j].type)
