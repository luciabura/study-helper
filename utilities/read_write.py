import os


def read_file(path):
    """Returns the textual context as a whole string from file
    @:param the file path you want to read from """
    FILE_PATH = path
    FILE_CONTENT = open(FILE_PATH, "r")
    FILE_TEXT = FILE_CONTENT.read()
    return FILE_TEXT


def print_to_file(content, path):
    file = open(path, "w")
    print(content, file=file)


def print_summary_to_file(file_text, input_path, output_dir, identifier=''):
    file_name = input_path.split('/').pop()
    name_parts = file_name.split('.')
    file_ending = '.' + name_parts.pop()
    file_name = '.'.join(name_parts)
    file_name = file_name.split('_')

    output_file_name = file_name[0]
    output_file_name += '_summary'
    output_file_name += identifier
    output_file_name += file_ending

    output_file_path = os.path.join(output_dir, output_file_name)

    print_to_file(file_text, output_file_path)
    print('Printed to: ' + str(output_file_path))
