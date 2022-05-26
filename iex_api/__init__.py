from dotenv import load_dotenv

load_dotenv()


class TableNotFoundError(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, message="Table was not found in the PostGres Database"):
        self.message = message
        super().__init__(self.message)


