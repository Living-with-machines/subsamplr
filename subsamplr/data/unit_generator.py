import getpass
import sqlalchemy as db  # type: ignore
import pandas as pd  # type: ignore
from urllib import parse


class UnitGenerator:
    """A generator of subsampling units."""

    @staticmethod
    def generate_units(df, unit_id, variables):
        """Generate subsampling units.

        Args:
            df (data frame): A table of unit data.
            unit_id   (str): The column name in df that contains unit identifiers.
            variables      : A collection of Variable instances.

        Returns:
            A sequence of pairs of the form (id, values), where id is a unit
            identifier and values is a list of corresponding values, one for
            each of the given variables.

        The data frame column names must include the unit_id and each of the
        variable names.
        """

        for index, row in df.iterrows():

            if not unit_id in df.columns.values:
                raise ValueError(
                    f"Unit ID column {unit_id} not found in data frame.")
            id = row[unit_id]

            for var in variables:
                if not var.name in df.columns.values:
                    raise ValueError(
                        f"Variable name {var.name} not found in data frame colums.")

            values = UnitGenerator.extract_values(
                row=row, unit_id=unit_id, variables=variables)

            # If any value is missing, discard this unit.
            if not values:
                continue

            yield id, tuple(values)

    @staticmethod
    def extract_values(row, unit_id, variables):

        # Extract the values for each variable from a data frame row.
        values = []
        for var in variables:

            if pd.isnull(row[var.name]):
                return None

            typed_value = var.type(row[var.name])
            if not typed_value == row[var.name]:
                raise Warning(
                    f"Type cast for unit {row[unit_id]} to {var.type} modified value: {row[var.name]}")
            values.append(typed_value)

        return values


class DbUnitGenerator(UnitGenerator):
    """A generator of subsampling units from a database connection."""

    def __init__(self, dialect, host, port, database, user):
        self.dialect = dialect
        self.host = host
        self.port = port
        self.database = database
        self.user = user

    def db_engine(self):

        password = getpass.getpass(
            prompt=f"\nPassword for database user {self.user}: ")

        host_prefix = self.host.partition('.')[0]
        connection_string = f'{self.dialect}://{self.user}@{host_prefix}:{parse.quote_plus(password)}@{self.host}:{self.port}/{self.database}'
        try:
            engine = db.create_engine(connection_string)
            return engine
        except Exception as e:
            print(f'Error: {e}')

    def fetch_data(self, query: str):
        """
        Fetches data from the database from which units can be constructed.

        Args:
            query (str): An SQL query.

        Returns:
            A pandas data frame.
        """

        engine = self.db_engine()
        connection = engine.connect()

        # TODO: use the 'unit' argument to select the query text.
        ResultProxy = connection.execute(query)

        # TODO: use the approach given in section 'Dealing with large ResultSet' at
        # https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91
        # to process results in batches (i.e. construct units ).
        ResultSet = ResultProxy.fetchall()
        df = pd.DataFrame(ResultSet)
        df.columns = ResultSet[0].keys()

        return df
