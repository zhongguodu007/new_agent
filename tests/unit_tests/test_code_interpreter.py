# -*- coding: utf-8 -*-
import sys

import pytest
sys.path.append('C:\\Users\\DELL\\Desktop\\rag\\langchain-glm')
from langchain_glm.agent_toolkits.all_tools.code_interpreter_tool import (
    CodeInterpreterAllToolExecutor,
)



# 3.9以上才能运行
@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires Python 3.9 or higher")
def test_python_ast_interpreter():
    out = CodeInterpreterAllToolExecutor._python_ast_interpreter(
        code_input='''
for i in range(10):
    print('Hello world!')
'''
    )
    tool_name = 'PythonExec'
    code_input = '''
for i in range(10):
    print('Hello world')
'''
    excutor = CodeInterpreterAllToolExecutor(platform_params={"sandbox": "none"}, name='python interpretor')
    result = excutor.run(
        tool=tool_name,
        tool_input=code_input,
        log='111'
    )
    print(result)
#     assert (
#         out.data
#         != """Access：code_interpreter,python_repl_ast, Message: print('Hello, World!')
# Hello, World!
# """
#     )

if __name__ == '__main__':
    test_python_ast_interpreter()