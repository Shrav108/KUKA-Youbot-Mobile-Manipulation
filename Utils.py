### There is a NaN in Matrix / Tensor Exception
class ThereIsNaN(Exception):
    """Class to raise that a NaN is in matrix or tensor."""
    def __init__(self, message = "There is a NaN value in the Matrix or Tensor."):
        super().__init__(message)
        self.message = message