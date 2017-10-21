class Log:
    def __init__(self):
        pass

    def write_log(self, log_context, log_file_name):
        with open(log_file_name, "a") as log_file:
            log_file.write(log_context)
