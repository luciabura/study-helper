def read_file(path):
    """Returns the textual context as a whole string from file
    @:param the file path you want to read from """
    FILE_PATH = path
    FILE_CONTENT = open(FILE_PATH, "r")
    FILE_TEXT = FILE_CONTENT.read()
    return FILE_TEXT

