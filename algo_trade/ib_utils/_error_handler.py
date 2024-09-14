import sys
import traceback
import os

# error handling
def show_exception_and_exit(exc_type, exc_value, tb_object):
    """Catches Exceptions and Prevents the App from Auto Closing"""
    traceback.print_exception(exc_type, exc_value, tb_object)

    os._exit(-1)

sys.excepthook = show_exception_and_exit