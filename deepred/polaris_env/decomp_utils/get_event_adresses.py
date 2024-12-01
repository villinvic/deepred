
def parse_asm_to_enum(asm_file_path, initial_offset):
    """
    Parses an ASM file to create an enum class of event names with hex values.

    :param asm_file_path: Path to the ASM file.
    :param initial_offset: Initial address offset (integer).
    :return: None (prints the enum class).
    """
    import re

    const_def_pattern = re.compile(r'^\s*const_def\s*$')
    event_pattern = re.compile(r'^\s*const\s+(EVENT_\w+)\s*$')
    skip_pattern = re.compile(r'^\s*const_skip\s+(\d*)\s*$')
    goto_pattern = re.compile(r'^\s*const_next\s+\$(\d+)\s*$')


    # Initialize state variables
    current_address = initial_offset
    current_bit = 0
    event_dict = {}

    try:
        with open(asm_file_path, 'r') as asm_file:
            for line in asm_file:
                # Detect the start of a new const_def block
                if const_def_pattern.match(line):
                    continue  # const_def lines are markers and don't affect addresses

                event_match = event_pattern.match(line)
                if event_match:
                    event_name = event_match.group(1)
                    event_dict[event_name] = (current_address, current_bit)
                    # Update bit pointer
                    current_bit += 1
                    if current_bit >= 8:  # Move to the next byte if the bit exceeds 7
                        current_bit = 0
                        current_address += 1
                    continue

                skip_match = skip_pattern.match(line)
                if skip_match:
                    try:
                        skip_count = int(skip_match.group(1))
                    except:
                        skip_count = 1

                    current_bit += skip_count

                    current_address += current_bit // 8  # Add full bytes if bits overflow
                    current_bit %= 8
                    continue

                goto_match = goto_pattern.match(line)
                if goto_match:
                    goto_bit = int(goto_match.group(1), 16)

                    current_address = initial_offset + goto_bit // 8
                    current_bit = goto_bit % 8
                    continue

        # Output the enum class
        print("from enum import Enum\n")
        print("class EventFlags(Enum):")
        for event_name, (address, bitmask) in event_dict.items():
            print(f"    {event_name} = {(address-initial_offset)*8 + bitmask} # (0x{address:02X}, 0x{bitmask:02X})")
        print("\n    def address(self):")
        print("        return self.value[0]")
        print("\n    def bitmask(self):")
        print("        return self.value[1]")

    except FileNotFoundError:
        print(f"Error: The file '{asm_file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# cf https://github.com/pret/pokered/commit/c9c59dc34323a5e0b1886db5f845c8d41620826e#diff-bae0b63c9fce8a284913eaaf988a93c5277810a6a156a8578873287147f32dc2L2562
asm_file_path = "event_consts.asm"
initial_offset = 0xD747
parse_asm_to_enum(asm_file_path, initial_offset)
