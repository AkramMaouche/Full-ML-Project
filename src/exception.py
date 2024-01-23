import sys 
from src.logger import logging

def error_message_detail(error,err_detail:sys): 

    _,_,exc_tb = err_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename()

    error_msg = " error occued name[{0}] line number[{1}] eror message [{2}]".format(
        file_name,exc_tb,str(error))
    
    return error_msg 

class CustomExeption(Exception): 
    def __init__(self,error_msg,error_detail): 
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg,err_detail = error_detail)

    def __str__(self): 
        return self.error_msg

