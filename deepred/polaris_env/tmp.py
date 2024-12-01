import csv
import enum
from enum import Enum

from deepred.polaris_env.enums import PKMN_RB_MAPS


def csv_to_enum(csv_file_path, enum_class_name):
    """
    Converts the first two rows of a CSV file into a Python Enum.

    Args:
        csv_file_path (str): Path to the CSV file.
        enum_class_name (str): The desired name for the Enum class.

    Returns:
        Enum: A Python Enum class dynamically created.
    """
    with open(csv_file_path, mode='r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t')  # Use tab as delimiter
        rows = list(reader)

        # Create a dictionary for the Enum
        enum_dict = {}
        print([len(r) for r in rows])
        for row in rows:
            poke_name = row[3].rstrip().replace("(", "").replace("-", "").replace(")", "").replace(" ", "_").replace(".", "")
            value = row[0]
            try:
                enum_dict[poke_name] = int(value, 16)  # Convert hex to integer
            except ValueError as e:
                enum_dict[poke_name] = enum.auto()

        # Dynamically create the Enum class
        return Enum(enum_class_name, enum_dict)


def dict_to_enum(d, enum_class_name):
    """
    Converts the first two rows of a CSV file into a Python Enum.

    Args:
        csv_file_path (str): Path to the CSV file.
        enum_class_name (str): The desired name for the Enum class.

    Returns:
        Enum: A Python Enum class dynamically created.
    """
    enum_dict = {}
    for value, map_name in d.items():
        poke_name = map_name.replace("(", "").replace("-", "").replace(")", "").replace(" ", "_").replace(".", "").replace("'", "")
        try:
            enum_dict[poke_name] = int(hex(value), 16)  # Convert hex to integer
        except ValueError as e:
            enum_dict[poke_name] = enum.auto()

    # Dynamically create the Enum class
    return Enum(enum_class_name, enum_dict)


def pretty_print_enum(enum_class):
    """
    Pretty-prints the enum in the format NAME = hex.

    Args:
        enum_class (Enum): The Enum class to pretty-print.
    """
    for member in enum_class:
        print(f"{member.name} = {hex(member.value)}".upper())


# Example usage
if __name__ == "__main__":
    # Specify your TSV file path here
    csv_file_path = "pokemons.csv"
    d = PKMN_RB_MAPS
    # Convert the CSV to an Enum class
    MyEnum = dict_to_enum(d, "Map")

    # Test the Enum class
    pretty_print_enum(MyEnum)
